#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import abc
import warnings

from tqdm import tqdm
from loguru import logger

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers.optimization import Adafactor, AdamW, get_scheduler

# this is used to supress an undesired warning emitted by pytorch versions 1.4.2-1.7.0
try:
    from torch.optim.lr_scheduler import SAVE_STATE_WARNING
except ImportError:
    SAVE_STATE_WARNING = ""


def reissue_pt_warnings(caught_warnings):
    # Reissue warnings that are not the SAVE_STATE_WARNING
    if len(caught_warnings) > 1:
        for w in caught_warnings:
            if w.category != UserWarning or w.message != SAVE_STATE_WARNING:
                warnings.warn(w.message, w.category)


def is_apex_available():
    import importlib.util
    return importlib.util.find_spec("apex") is not None


if is_apex_available():
    from apex import amp


class Trainer:
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        #  data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        optimizers: Tuple[torch.optim.Optimizer,
                          torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        self.model = model
        self.args = args
        #  self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers

        if hasattr(model, "is_parallelizable"
                   ) and model.is_parallelizable and model.model_parallel:
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway
        if not (self.is_model_parallel or args.deepspeed):
            model = model.to(args.device)

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        if self.is_master_process():
            os.makedirs(self.args.output_dir, exist_ok=True)

        if args.max_steps > 0:
            logger.info(
                "max_steps is given, it will override any value given in num_train_epochs"
            )
        # Enforce rules on using datasets with no __len__
        if train_dataset is not None and not isinstance(
                train_dataset, abc.Sized) and args.max_steps <= 0:
            raise ValueError(
                "train_dataset does not implement __len__, max_steps has to be specified"
            )
        if eval_dataset is not None and not isinstance(eval_dataset,
                                                       abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        # Mixed precision setup
        self.use_apex = False
        self.use_amp = False
        self.fp16_backend = None

        from torch.cuda.amp import autocast
        if args.fp16:
            if args.fp16_backend == "auto":
                self.fp16_backend = "amp"
            else:
                self.fp16_backend = args.fp16_backend
            logger.info(f"Using {self.fp16_backend} fp16 backend")

        if args.fp16 and not args.deepspeed:  # deepspeed manages its own fp16
            if self.fp16_backend == "amp":
                self.use_amp = True
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                if not is_apex_available():
                    raise ImportError(
                        "Using FP16 with APEX but APEX is not installed, please refer to https://www.github.com/nvidia/apex."
                    )
                self.use_apex = True

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(
                epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    0.0,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {
                    "scale_parameter": False,
                    "relative_step": False
                }
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_dpp:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters,
                                               **optimizer_kwargs)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if os.path.isfile(os.path.join(
                checkpoint, "optimizer.pt")) and os.path.isfile(
                    os.path.join(checkpoint, "scheduler.pt")):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint, "optimizer.pt"),
                           map_location=self.args.device))
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(
                    torch.load(os.path.join(checkpoint, "scheduler.pt")))
            reissue_pt_warnings(caught_warnings)

        if self.deepspeed:
            # Not sure how to check if there is a saved deepspeed checkpoint, but since it just return None if it fails to find a deepspeed checkpoint this is sort of a check-n-load function
            self.deepspeed.load_checkpoint(checkpoint,
                                           load_optimizer_states=True,
                                           load_lr_scheduler_states=True)

    def is_master_process(self) -> bool:
        return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset,
                      torch.utils.data.IterableDataset) or not isinstance(
                          self.train_dataset, collections.abc.Sized):
            return None

        # Gather the number of processes and this process index.
        if (self.args.parallel_mode == ParallelMode.DISTRIBUTED or
                self.args.parallel_mode == ParallelMode.SAGEMAKER_DISTRIBUTED):
            num_processes = torch.distributed.get_world_size()
            process_index = torch.distributed.get_rank()
        else:
            num_processes = 1
            process_index = 0

        # Build the sampler.
        if num_processes <= 1:
            return RandomSampler(self.train_dataset)
        else:
            return DistributedSampler(self.train_dataset,
                                      num_replicas=num_processes,
                                      rank=process_index)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        **kwargs,
    ):
        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )

        model_reloaded = False
        # Load potential model checkpoint
        PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, PYTORCH_WEIGHTS_NAME)):
            logger.info(f"Loading model from {resume_from_checkpoint}).")
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(resume_from_checkpoint)
                model_reloaded = True
            else:
                state_dict = torch.load(
                    os.path.join(resume_from_checkpoint, PYTORCH_WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if not self.is_model_parallel:
                self.model = self.model.to(self.args.device)
            #  self.model_wrapped = self.model

        train_dataloader = self.get_train_dataloader()

        train_dataset_is_sized = isinstance(self.train_dataset,
                                            collections.abc.Sized)
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(
                train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0)
            else:
                max_steps = math.ceil(self.args.num_train_epochs *
                                      num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
