#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import random
from copy import deepcopy
from datetime import datetime
from typing import Type, Union

import dill
import numpy as np
import pytorch_lightning as pl
import torch
import torch.functional as F
import torch.nn as nn
from loguru import logger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from theta.nlp.arguments import (DataArguments, ModelArguments, TaskArguments,
                                 TrainingArguments,
                                 create_instance_from_arguments,
                                 generate_method_kwargs_from_arguments)
from transformers import (AdamW, AutoConfig, AutoModel, AutoTokenizer,
                          BertForSequenceClassification, ElectraConfig,
                          XLNetConfig)
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup)

from ...utils import seed_everything

os.environ['TOKENIZERS_PARALLELISM'] = "true"


# ------------------------------ Dataset ------------------------------
class BaseDataset(object):
    """
        Dataset负责完成传入模型数据的编码工作
        编码data_generator生成的数据
    """
    def __init__(self, data_args, data_generator, label2id, tokenizer):
        super(BaseDataset, self).__init__()
        self.data_args = data_args
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.sids = []
        self.seg_spans = []

        self.encoded_data_list = []
        for x in tqdm(data_generator(), desc="Encoding"):
            encoded = self._encode_item(x)
            if isinstance(encoded, list):
                self.encoded_data_list.extend(encoded)
                sids = [f"{x[0]}-{i}" for i in range(len(encoded))]
                self.sids.extend(sids)
            else:
                self.encoded_data_list.append(encoded)
                self.sids.append(x[0])

    def _encode_item(self, x):
        raise NotImplementedError

    def __iter__(self):
        for x in self.encoded_data_list:
            yield x

    def __getitem__(self, idx):
        return self.encoded_data_list[idx]

    def __len__(self):
        return len(self.encoded_data_list)


# ------------------------------ TaskData ------------------------------
class TaskData:
    """
    TaskData负责完成Samples -> Dataset的转换
    对模型提供train_dataset、val_dataset、test_dataset
    """
    def __init__(self,
                 data_args: Type[DataArguments],
                 train_samples,
                 test_samples,
                 val_samples=None,
                 tokenizer=None):
        self.data_args = data_args
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.val_samples = val_samples
        self.tokenizer = tokenizer

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        if train_samples is not None and test_samples is not None:
            assert train_samples.label2id == test_samples.label2id
        if train_samples is not None:
            self.label2id = train_samples.label2id
        elif test_samples is not None:
            self.label2id = test_samples.label2id
        else:
            raise ValueError(
                f"Either train_samples or test_samples must be not None.")

    def load_train_data(self):
        self.train_samples.load_samples()

        if self.val_samples is not None:
            selv.val_samples.load_samples()
            self._splitted_train_samples = self.train_samples
            self._splitted_val_samples = self.val_samples
        else:
            all_train_samples = self.train_samples.shuffle()
            self._splitted_train_samples, self._splitted_val_samples = all_train_samples.split(
                ratios=self.data_args.split_ratios,
                random_state=self.data_args.random_state)
            logger.info(f"total train samples: {len(self.train_samples)}")
        logger.info(f"train samples: {len(self._splitted_train_samples)}")
        logger.info(f"val samples: {len(self._splitted_val_samples)}")

    def load_test_data(self):
        self.test_samples.load_samples()
        logger.info(f"test samples: {len(self.test_samples)}")

    def cache_dataset(self, mode, build_dataset_fn):
        cache_file = f"{self.data_args.cache_dir}/{mode}_dataset.cache"
        if not self.data_args.overwrite_cache and os.path.exists(cache_file):
            logger.info(f"Load cached {mode} dataset from {cache_file}")
            dataset = dill.load(open(cache_file, 'rb'))
            logger.warning(f"{len(dataset)} lines loaded.")
        else:
            dataset = build_dataset_fn()
            dill.dump(dataset, open(cache_file, 'wb'))
            logger.info(
                f"Save cached {mode} dataset {len(dataset)} lines to {cache_file}"
            )
        return dataset

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = self.cache_dataset("train",
                                                     self.build_train_dataset)
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            self._val_dataset = self.cache_dataset("val",
                                                   self.build_val_dataset)
        return self._val_dataset

    @property
    def test_dataset(self):
        if self._test_dataset is None:
            self._test_dataset = self.cache_dataset("test",
                                                    self.build_test_dataset)
        return self._test_dataset


def save_args(args, model_path):
    os.makedirs(model_path, exist_ok=True)
    args_file = os.path.join(model_path, "training_args.json")
    logger.info(f"Save args in {args_file}")
    json.dump(
        {
            k: v
            for k, v in args.__dict__.items() if v is None
            or type(v) in [bool, str, int, float, dict, list, tuple]
        },
        open(args_file, 'w'),
        ensure_ascii=False,
        indent=2)


class TransformerModel(nn.Module):
    def __init__(self,
                 model_name_or_path,
                 tokenizer=None,
                 automodel_cls=AutoModel):
        super(TransformerModel, self).__init__()
        assert automodel_cls is not None
        self.automodel_cls = automodel_cls
        logger.info(f"automodel_cls: {self.automodel_cls}")
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        logger.info(f"{self.config}")

        if tokenizer is None:
            #  if model_name_or_path not in ['clue/roberta_chinese_pair_large']:
            #      tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
            #                                                fast=True)
            #  else:
            #      # self.config.vocab_size == 21128
            #      tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",
            #                                                fast=True)
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",
                                                      fast=True)
        self.tokenizer = tokenizer

        self.model_name_or_path = model_name_or_path
        #  self.transformer = self.automodel_cls.from_config(self.config)
        self.load_from_config()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights
        Derived from BertPreTrainedModel._init_weights() in modeling_bert.py of transformers.
        """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    #  @staticmethod
    #  def _init_weights(blocks, **kwargs):
    #      """
    #      参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
    #      """
    #      for block in blocks:
    #          for module in block.modules():
    #              if isinstance(module, nn.Linear):
    #                  if module.bias is not None:
    #                      nn.init.zeros_(module.bias)
    #              elif isinstance(module, nn.Embedding):
    #                  nn.init.normal_(module.weight,
    #                                  mean=0,
    #                                  std=kwargs.pop('initializer_range', 0.02))
    #              elif isinstance(module, nn.LayerNorm):
    #                  nn.init.ones_(module.weight)
    #                  nn.init.zeros_(module.bias)

    def _is_xlnet(self):
        if isinstance(self.config, XLNetConfig):
            return True
        else:
            return False

    #  def load_from_pretrained(self, model_path):
    #      assert self.automodel_cls is not None
    #      self.transformer = self.automodel_cls.from_pretrained(model_path)
    #
    #      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #      self.transformer.to(device)

    #  def _load_from_pretrained(self, model_path, automodel_cls=None):
    #      if automodel_cls is not None:
    #          self.transformer = automodel_cls.from_pretrained(model_path)
    #      else:
    #          self.transformer = AutoModel.from_pretrained(model_path)
    #          #  self.transformer = BertForSequenceClassification.from_pretrained(model_path)
    #
    #      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #      self.transformer.to(device)

    def save_model(self, model_path):
        os.makedirs(model_path, exist_ok=True)
        #  model_to_save = (self.transformer.module if hasattr(
        #      self.transformer, "module") else self.transformer)
        #  model_to_save.save_pretrained(model_path)

        self.config.save_pretrained(os.path.abspath(model_path) + '/')
        self.tokenizer.save_vocabulary(os.path.abspath(model_path) + '/')

    #  def train(self):
    #      self.transformer.train()
    #
    #  def eval(self):
    #      self.transformer.eval()
    #
    #  def __call__(self, *args, **kwargs):
    #      outputs = self.transformer(*args, **kwargs)
    #      return outputs

    def forward(self, *args, **kwargs):
        outputs = self.transformer(*args, **kwargs)
        return outputs


# ------------------------------ TaskRunner ------------------------------
class TaskRunner(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(TaskRunner, self).__init__()
        self.save_hyperparameters()

        #  model_name_or_path = self.hparams.model_name_or_path
        #  self.tokenizer = tokenizer

        #  if model_name_or_path is None:
        #      model_name_or_path = os.path.join(self.hparams.task_dir,
        #                                        "checkpoint")
        #  self.transformer_model = TransformerModel(model_name_or_path,
        #                                            tokenizer)

        self.warmup_steps = None
        self.total_steps = None
        self.wait_count = 0

        self.best_score = 0.0 if self.hparams.greater_is_better else float(
            'inf')

    #  @property
    #  def tokenizer(self):
    #      return self.transformer_model.tokenizer

    def _is_xlnet(self):
        config = self.transformer_model.config
        if isinstance(config, XLNetConfig):
            return True
        else:
            return False
        #  return 'XLNetLMHeadModel' in config.architectures
    def _is_electra(self):
        if isinstance(self.transformer_model.config, ElectraConfig):
            return True
        else:
            return False

    #  def load_model(self, model_path):
    #      pl_model_checkpoint_file = f"{model_path}/pl_model.ckpt"
    #      if os.path.exists(pl_model_checkpoint_file):
    #          logger.info(f"Load PL module from {pl_model_checkpoint_file}.")
    #          checkpoint = dill.load(open(pl_model_checkpoint_file, 'rb'))
    #          self.load_state_dict(checkpoint['state_dict'])
    #      else:
    #          #  self.model.load_model(model_path)
    #          logger.info(f"Load transformer model from {model_path}")
    #          self.model.load_model(model_path)

    def load_from_checkpoint(self, model_path):
        #  self.model.load_from_config()
        if os.path.isdir(model_path):
            pl_model_checkpoint_file = f"{model_path}/pl_model.ckpt"
        else:
            pl_model_checkpoint_file = model_path
        if os.path.exists(pl_model_checkpoint_file):
            logger.info(f"Load PL module from {pl_model_checkpoint_file}.")
            checkpoint = dill.load(open(pl_model_checkpoint_file, 'rb'))
            self.load_state_dict(checkpoint['state_dict'])
        else:
            logger.error(
                f"Checkpoint file {pl_model_checkpoint_file} does not exists.")

    def load_from_pretrained(self, model_name_or_path):
        logger.info(f"Load from pretrained model by {model_name_or_path}")
        self.model.load_from_pretrained(model_name_or_path)

    def save_model(self, model_path):
        os.makedirs(model_path, exist_ok=True)

        pl_model_checkpoint_file = f"{model_path}/pl_model.ckpt"

        # FIXME
        dill.dump({'state_dict': self.state_dict()},
                  open(pl_model_checkpoint_file, 'wb'))
        logger.warning(f"Save PL module in {pl_model_checkpoint_file}")

        self.model.save_model(model_path)
        logger.warning(f"Save transformer model in {model_path}")

    def configure_optimizers(self):
        #  param_optimizer = list(self.transformer_model.model.named_parameters())
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate":
                self.hparams.weight_decay
            },
            {
                "params": [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate":
                0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate)  #,
        #  correct_bias=False)

        #  return optimizer

        assert self.warmup_steps is not None
        assert self.total_steps is not None

        # Inspired from https://github.com/PyTorchLightning/pytorch-lightning/issues/328
        #  def lr_exp_warmup(steps):
        #      if steps < self.warmup_steps:
        #          lr_scale = 0.1**(self.warmup_steps - steps)
        #      else:
        #          lr_scale = 0.95**steps
        #
        #      return lr_scale
        #
        #  scheduler = LambdaLR(optimizer, lr_lambda=lr_exp_warmup)

        schedule_fn = get_linear_schedule_with_warmup
        #  schedule_fn = get_cosine_schedule_with_warmup
        #  schedule_fn = get_cosine_with_hard_restarts_schedule_with_warmup
        scheduler = schedule_fn(optimizer,
                                num_warmup_steps=self.warmup_steps,
                                num_training_steps=self.total_steps)

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return ([optimizer], [scheduler_dict])

    #  def training_step_end(self, batch_parts_outputs):
    #      if self.trainer.global_step > 0 and self.trainer.global_step % self.hparams.eval_steps == 0:
    #          self.trainer.run_evaluation()

    @property
    def current_epoch(self):
        return self.trainer.current_epoch

    @property
    def max_epochs(self):
        return self.trainer.max_epochs

    #  def on_train_start(self):
    #      self.model.train()
    #
    #  def on_validation_start(self, model):
    #      self.model.eval()
    #
    #  def on_validation_end(self, model):
    #      self.model.train()
    #
    #  def on_test_start(self, model):
    #      self.model.eval()
    #
    #  def on_test_end(self, model):
    #      self.model.train()

    def save_best_model(self, eval_outputs: dict):

        if 'val_loss' in eval_outputs:
            val_loss = eval_outputs['val_loss']
            logger.info(
                f"Epoch {self.current_epoch}/{self.max_epochs} val_loss: {val_loss:.4f}"
            )
        else:
            logger.warning(f"No val_loss in eval_outputs: {eval_outputs}")

        if self.hparams.metric_for_best_model in eval_outputs:
            curr_score = eval_outputs[self.hparams.metric_for_best_model]

            is_best = False
            if self.hparams.greater_is_better:
                if curr_score > self.best_score:
                    is_best = True
            else:
                if curr_score < self.best_score:
                    is_best = True

            if is_best:
                self.wait_count = 0
                logger.warning(
                    f"Best {self.hparams.metric_for_best_model}: {curr_score:.4f} / {self.best_score:.4f}"
                )
                self.best_score = curr_score

                self.save_model(
                    os.path.join(self.hparams.task_dir, "checkpoint"))
            else:
                self.wait_count += 1
                logger.info(
                    f"{self.hparams.metric_for_best_model}: {curr_score:.4f} / {self.best_score:.4f}, earlystopping_patience: {self.wait_count}/{self.hparams.earlystopping_patience}"
                )
                if self.wait_count >= self.hparams.earlystopping_patience:
                    self.trainer.should_stop = True


# ------------------------------ Task ------------------------------


class BaseTask():
    """
    实验任务(ExperimentTask)
    定义一次任务中必须包含的数据、模型、参数。
    支持任务执行、加载复现。

    task_args = TaskArguments.parse_args()
    task_data = ExpData(data_args=task_args.data_args)
    task_trainer = ExpModel(model_args=task_args.model_args,
                          num_labels=len(glue_labels))

    task = ExpTask(args=task_args,
                   data=task_data,
                   trainer=task_trainer)

    task.execute(glue_labels)

    """
    def __init__(self, args: Type[TaskArguments], data: Type[TaskData],
                 runner: Type[TaskRunner]):
        self.args = args
        self.data = data
        self.runner = runner

        os.environ['TOKENIZERS_PARALLELISM'] = "true"
        seed_everything(self.args.training_args.seed)  #固定随机种子

        #  self.args.training_args.device = torch.device(
        #      'cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def rootdir(self):
        return self.args.rootdir

    @property
    def data_args(self):
        return self.args.data_args

    @property
    def model_args(self):
        return self.args.model_args

    @property
    def training_args(self):
        return self.args.training_args

    @property
    def remaining_args(self):
        return self.args.remaining_args

    @property
    def latest_path(self):
        return self.training_args.latest_path

    @property
    def checkpoint_path(self):
        return self.model_args.checkpoint_path

    @property
    def test_results_file(self):
        return os.path.join(self.training_args.task_dir, "test_results.pkl")

    def load_test_results(self, test_results_file=None):
        if test_results_file is None:
            test_results_file = os.path.join(self.latest_path,
                                             "test_results.pkl")
        logger.info(f"Load test result from {test_results_file}")
        test_results = dill.load(open(test_results_file, 'rb'))
        return test_results

    def dump_test_results(self, test_results, test_results_file=None):
        if test_results_file is None:
            test_results_file = self.test_results_file
        dill.dump(test_results, open(test_results_file, 'wb'))
        logger.info(f"Dump test results to {test_results_file}.")

        return test_results_file

    @property
    def train_dataloader(self):
        train_dataset = self.data.train_dataset
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}.")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,
            num_workers=8)
        return train_dataloader

    @property
    def val_dataloader(self):
        val_dataset = self.data.val_dataset
        for index in random.sample(range(len(val_dataset)), 1):
            logger.info(
                f"Sample {index} of the val set: {val_dataset[index]}.")

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True,
            num_workers=8)
        return val_dataloader

    @property
    def test_dataloader(self):
        test_dataset = self.data.test_dataset
        for index in random.sample(range(len(test_dataset)), 1):
            logger.info(
                f"Sample {index} of the test set: {test_dataset[index]}.")

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.training_args.per_device_test_batch_size,
            collate_fn=test_dataset.collate_fn,
            pin_memory=True,
            num_workers=8)
        return test_dataloader

    def get_latest_submission_file(self, ext="json", prefix="submission"):
        ts = datetime.now().strftime("%Y%m%d.%H%M%S")
        real_path = os.readlink(self.latest_path)
        task_id = os.path.basename(real_path)
        submission_file = os.path.join(self.training_args.submissions_dir,
                                       f"{prefix}_{task_id}_{ts}.{ext}")
        return submission_file

    def generate_submission(self):
        logger.warning(
            f"Call generate_submission() implemented in base class Task.")

    def execute(self, *args, **kwargs):

        model_args = self.model_args
        data_args = self.data_args
        training_args = self.training_args
        remaining_args = self.remaining_args

        return_dict = {}

        # ------------------------------ do_train ------------------------------
        if training_args.do_train:
            model_path = model_args.model_name_or_path
            self.runner.load_from_pretrained(model_path)
            self.data.tokenizer = self.runner.model.tokenizer
            self.data.load_train_data()

            def setup_warmup_steps():
                epoch_steps = int(
                    len(self.data.train_dataset) /
                    training_args.per_device_train_batch_size)

                warmup_method = training_args.warmup_method
                if warmup_method == 'auto':
                    if training_args.max_steps is not None:
                        training_args.warmup_steps = int(
                            training_args.max_steps / 10)
                    else:
                        training_args.warmup_steps = int(
                            epoch_steps * training_args.max_epochs / 10)
                elif warmup_method == 'by_epoch':
                    training_args.warmup_steps = epoch_steps * training_args.warmup_epochs
                elif warmup_method == 'by_rate':
                    if training_args.max_steps is not None:
                        training_args.warmup_steps = int(
                            training_args.max_steps *
                            training_args.warmup_rate)
                    else:
                        training_args.warmup_steps = int(
                            epoch_steps * training_args.max_epochs *
                            training_args.warmup_rate)
                elif warmup_method == 'by_steps':
                    if training_args.warmup_steps > 0:
                        pass
                    else:
                        training_args.warmup_steps = epoch_steps * training_args.warmup_epochs

            setup_warmup_steps()

            max_epochs = self.training_args.max_epochs
            if max_epochs:
                total_steps = int(
                    len(self.data.train_dataset) / self.training_args.
                    per_device_train_batch_size) * max_epochs
            else:
                total_steps = self.training_args.max_steps

            self.runner.warmup_steps = training_args.warmup_steps
            self.runner.total_steps = total_steps

            #  self.training_args.max_steps = total_steps

            logger.warning(f"warmup_steps: {self.training_args.warmup_steps}")
            logger.warning(f"total_steps: {total_steps}")

            val_epoch_steps = int(
                len(self.data.val_dataset) /
                training_args.per_device_eval_batch_size)

            #  trainer = create_instance_from_arguments(
            #      pl.Trainer, self.training_args.to_dict())
            trainer_kwargs = generate_method_kwargs_from_arguments(
                pl.Trainer, method="__init__", args=training_args.to_dict())

            trainer_kwargs['precision'] = 16 if training_args.fp16 else 32

            if training_args.eval_steps_or_interval >= 10:
                trainer_kwargs[
                    'val_check_interval'] = training_args.eval_steps_or_interval / epoch_steps
            else:
                trainer_kwargs[
                    'val_check_interval'] = training_args.eval_steps_or_interval

            trainer_kwargs['checkpoint_callback'] = False
            trainer_kwargs['num_sanity_val_steps'] = int(val_epoch_steps * 0.5)

            def symlink_latest_path(latest_path):
                if os.path.islink(latest_path):
                    os.remove(latest_path)
                os.symlink(os.path.basename(training_args.task_dir),
                           latest_path)

            symlink_latest_path(self.latest_path)

            def save_trainer_kwargs(save_dir):
                trainer_kwargs_file = os.path.join(save_dir,
                                                   "trainer_kwargs.json")
                json.dump(trainer_kwargs,
                          open(trainer_kwargs_file, 'w'),
                          ensure_ascii=False,
                          indent=2)
                logger.info(f"Saved trainer_kwargs to {trainer_kwargs_file}")

            save_trainer_kwargs(self.latest_path)

            #  if training_args.earlystopping_patience > 0:
            #      earlystopping_cb = EarlyStopping(
            #          monitor=training_args.metric_for_best_model,
            #          patience=training_args.earlystopping_patience)
            #      trainer_kwargs['callbacks'] = [earlystopping_cb]
            #
            logger.warning(f"trainer_kwargs: {trainer_kwargs}")

            trainer = pl.Trainer(**trainer_kwargs)
            trainer.fit(self.runner, self.train_dataloader,
                        self.val_dataloader)

        # ------------------------------ do_eval ------------------------------
        if training_args.do_eval:
            self.runner.load_from_checkpoint(self.checkpoint_path)
            self.data.tokenizer = self.runner.model.tokenizer
            self.data.load_train_data()

            val_dataloader = self.val_dataloader
            #  eval_results = do_eval(trainer, val_dataset, data_args)

        # ------------------------------ do_predict ------------------------------
        if training_args.do_predict:
            self.runner.load_from_checkpoint(self.checkpoint_path)
            self.data.tokenizer = self.runner.model.tokenizer
            self.data.load_test_data()

            test_dataloader = self.test_dataloader

            trainer_kwargs = generate_method_kwargs_from_arguments(
                pl.Trainer,
                method="__init__",
                args=self.training_args.to_dict())

            trainer_kwargs['precision'] = 16 if self.training_args.fp16 else 32

            trainer = pl.Trainer(**trainer_kwargs)

            trainer.test(self.runner, test_dataloaders=test_dataloader)

            test_results_file = self.dump_test_results(
                self.runner.test_results)

            return_dict['test_results_file'] = test_results_file

        if training_args.do_submit:
            submission_file = self.generate_submission()

            return_dict['submission_file'] = submission_file

        return return_dict
