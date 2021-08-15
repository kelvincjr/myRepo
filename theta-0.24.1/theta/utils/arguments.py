#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple, Type, Union

from loguru import logger

from transformers import HfArgumentParser
from transformers import TrainingArguments as HfTrainingArguments


def compare_arguments(one_args, another_args, return_identical=False):
    #  assert isinstance(
    #      another_args, BaseArguments
    #  ), f"another_args {type(another_args)} must be subtype of BaseArguments."

    clsname = one_args.__class__.__name__

    diff_attrs = []
    identical_attrs = []
    left_only_attrs = []
    right_only_attrs = []
    for k, v in one_args.__dict__.items():
        if k not in another_args.__dict__:
            left_only_attrs.append((clsname, k, v))
        else:
            v1 = getattr(another_args, k)
            if v == v1:
                if return_identical:
                    identical_attrs.append((clsname, k, v))
            else:
                if isinstance(v, BaseArguments):
                    comp_result = v.compare(v1, return_identical)
                    diff_attrs.extend(comp_result['diff'])
                    left_only_attrs.extend(comp_result['left_only'])
                    right_only_attrs.extend(comp_result['right_only'])
                    if return_identical:
                        identical_attrs.extend(comp_result['identical'])
                else:
                    diff_attrs.append((clsname, k, v, v1))

    for k, v in another_args.__dict__.items():
        if k not in one_args.__dict__:
            right_only_attrs.append((another_args.__class__.__name__, k, v))

    comp_result = {
        'diff': diff_attrs,
        'left_only': left_only_attrs,
        'right_only': right_only_attrs,
    }
    if return_identical:
        comp_result['identical'] = identical_attrs

    return comp_result


@dataclass
class DataArguments():
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: "},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. "
            "Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path of train file."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path of test file."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path of validation file."},
    )
    pad_to_max_length: Optional[bool] = field(
        default=True, metadata={"help": "Pad to max length"})

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": ("Overwrite the content of the cache directory.")},
    )
    training_samples_rate: float = field(
        default=1.0,
        metadata={"help": "How many rate of train samples to be used."},
    )

    #  split_ratio: Optional[Union[float, List, Tuple]] = field(
    #      default=0.9,
    #      #  default_factory=list,
    #      metadata={
    #          "help":
    #          "Split ratio of train/eval/test dataset. float for train or [train, eval, test]"
    #      },
    #  )

    def __post_init__(self):
        pass


@dataclass
class ModelArguments():
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Model name or path."},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })
    cache_dir: str = field(
        default=None,
        metadata={"help": "Cache dir"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help":
            "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help":
            "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help":
            "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        pass


@dataclass
class TrainingArguments(HfTrainingArguments):

    theta_version: Optional[str] = field(
        default="0.24",
        metadata={"help": "Theta version."},
    )
    #  output_dir: Optional[str] = field(
    #      default=None,
    #      metadata={
    #          "help":
    #          "The output directory where the model predictions and checkpoints will be written."
    #      },
    #  )
    #  overwrite_output_dir: bool = field(
    #      default=False,
    #      metadata={
    #          "help":
    #          ("Overwrite the content of the output directory."
    #           "Use this to continue training if output_dir points to a checkpoint directory."
    #           )
    #      },
    #  )
    #  run_name: Optional[str] = field(
    #      default=None,
    #      metadata={
    #          "help":
    #          "An optional descriptor for the run. Notably used for wandb logging."
    #      },
    #  )
    #
    #  learning_rate: float = field(
    #      default=2e-5,
    #      metadata={"help": "The initial learning rate for AdamW."},
    #  )
    #
    #  weight_decay: float = field(
    #      default=0.0,
    #      metadata={"help": "Weight decay for AdamW if we apply some."},
    #  )
    #
    #  adam_epsilon: float = field(
    #      default=1e-8,
    #      metadata={"help": "Epsilon for AdamW optimizer."},
    #  )
    #
    #  num_train_epochs: float = field(
    #      default=3.0,
    #      metadata={"help": "Total number of training epochs to perform."},
    #  )
    #  max_steps: int = field(
    #      default=-1,
    #      metadata={
    #          "help":
    #          "If > 0: set total number of training steps to perform. Override num_train_epochs."
    #      },
    #  )
    #
    #  warmup_steps: int = field(
    #      default=0,
    #      metadata={"help": "Linear warmup over warmup_steps."},
    #  )
    #
    #  lr_scheduler_type: str = field(
    #      default="linear",
    #      metadata={
    #          "help":
    #          "The scheduler type to use. One of SchedulerType "
    #          "['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']"
    #      },
    #  )
    #
    #  seed: int = field(
    #      default=42,
    #      metadata={
    #          "help":
    #          "Random seed that will be set at the beginning of training."
    #      },
    #  )
    #
    #  fp16: bool = field(
    #      default=False,
    #      metadata={
    #          "help":
    #          "Whether to use 16-bit (mixed) precision (through NVIDIA Apex) instead of 32-bit"
    #      },
    #  )
    #
    #  per_device_train_batch_size: int = field(
    #      default=8,
    #      metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    #  )
    #
    #  per_device_eval_batch_size: int = field(
    #      default=8,
    #      metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."},
    #  )
    #
    #  gradient_accumulation_steps: int = field(
    #      default=1,
    #      metadata={
    #          "help":
    #          "Number of updates steps to accumulate before performing a backward/update pass."
    #      },
    #  )
    #
    #  label_smoothing_factor: float = field(
    #      default=0.0,
    #      metadata={
    #          "help":
    #          "The label smoothing epsilon to apply (zero means no label smoothing)."
    #      },
    #  )
    #
    #  adafactor: bool = field(
    #      default=False,
    #      metadata={"help": "Whether or not to replace AdamW by Adafactor."},
    #  )
    #
    #  group_by_length: bool = field(
    #      default=False,
    #      metadata={
    #          "help":
    #          "Whether or not to group samples of roughly the same length together when batching."
    #      },
    #  )
    #
    #  logging_steps: int = field(
    #      default=500,
    #      metadata={"help": "Log every X updates steps."},
    #  )
    #
    #  evaluation_strategy: str = field(
    #      default="no",
    #      metadata={
    #          "help":
    #          "The evaluation strategy to use. One of EvaluationStrategy ['no', 'steps', 'epochs']."
    #      },
    #  )
    #
    #  eval_steps: int = field(
    #      default=None,
    #      metadata={"help": "Run an evaluation every X steps."},
    #  )
    #
    #  save_steps: int = field(
    #      default=500,
    #      metadata={"help": "Save checkpoint every X updates steps."},
    #  )
    #
    #  save_total_limit: Optional[int] = field(
    #      default=None,
    #      metadata={
    #          "help":
    #          ("Limit the total amount of checkpoints."
    #           "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
    #           )
    #      },
    #  )
    #
    #  load_best_model_at_end: Optional[bool] = field(
    #      default=False,
    #      metadata={
    #          "help":
    #          "Whether or not to load the best model found during training at the end of training."
    #      },
    #  )
    #  metric_for_best_model: Optional[str] = field(
    #      default=None,
    #      metadata={
    #          "help": "The metric to use to compare two different models."
    #      })
    #  greater_is_better: Optional[bool] = field(
    #      default=None,
    #      metadata={
    #          "help":
    #          "Whether the `metric_for_best_model` should be maximized or not."
    #      })
    #
    #  do_train: bool = field(default=False,
    #                         metadata={"help": "Whether to run training."})
    #  do_eval: bool = field(
    #      default=False,
    #      metadata={"help": "Whether to run eval on the dev set."})
    #  do_predict: bool = field(
    #      default=False,
    #      metadata={"help": "Whether to run predictions on the test set."})

    #  def __post_init__(self):
    #      pass


@dataclass
class TaskArguments():
    rootdir: str = "outputs"
    data_args: DataArguments = None
    model_args: ModelArguments = None
    training_args: TrainingArguments = None
    remaining_args: list = field(default_factory=list)

    #  data_args: DataArguments = field(
    #      default=DataArguments(),
    #      metadata={"help": "Arguments of data."},
    #  )
    #  model_args: ModelArguments = field(
    #      default=ModelArguments(),
    #      metadata={"help": "Arguments of model."},
    #  )
    #  training_args: TrainingArguments = field(
    #      default=TrainingArguments(),
    #      metadata={"help": "Arguments of training."},
    #  )
    #  remaining_args: list = field(
    #      default_factory=list,
    #      metadata={"help": "Remaining arguments come from command-line."},
    #  )

    def __post_init__(self):
        pass

    @classmethod
    def parse_args(cls):
        parser = HfArgumentParser(
            (DataArguments, ModelArguments, TrainingArguments))
        #  HfTrainingArguments))  #TrainingArguments))
        remaining_args = None
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            data_args, model_args, training_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1]))
        else:
            data_args, model_args, training_args, remaining_args = parser.parse_args_into_dataclasses(
                return_remaining_strings=True)
        task_args = TaskArguments(data_args=data_args,
                                  model_args=model_args,
                                  training_args=training_args,
                                  remaining_args=remaining_args)
        return task_args

    @classmethod
    def merge_args(cls, args, task_args):
        from dataclasses import asdict
        for k, v in asdict(task_args).items():
            setattr(args, k, v)
        return args


if __name__ == '__main__':
    task_args = TaskArguments.parse_args()
    logger.debug(f"{task_args}")
    logger.info(f"{asdict(task_args)}")

    default_args = TaskArguments()
    comp_result = task_args.compare(default_args)
    logger.warning(f"{comp_result}")
