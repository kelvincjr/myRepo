#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, shutil, random, json
from dataclasses import dataclass, field
from typing import List
from tqdm import tqdm
from loguru import logger
#  import mlflow


class Params:
    def log(self):
        pass
        #  for k, v in self.__dict__.items():
        #      if isinstance(v, Params):
        #          v.log()
        #      else:
        #          mlflow.log_param(k, v)

    def debug(self):
        for k, v in sorted(self.__dict__.items()):
            if isinstance(v, Params):
                v.debug()
            else:
                logger.debug(f"{k}: {v}")

    def update_parser(self, parser):
        for k, v in self.__dict__.items():
            if isinstance(v, Params):
                v.update_parser(parser)
            else:
                parser.set_defaults(**{k: v})
        return parser

    def update_args(self, args):
        for k, v in self.__dict__.items():
            if isinstance(v, Params):
                v.update_args(args)
            else:
                setattr(args, k, v)

        return args


@dataclass
class CommonParams(Params):
    dataset_name: str = None
    experiment_name: str = None
    tracking_uri: str = None
    artifact_path: str = None
    train_file: str = None
    eval_file: str = None
    test_file: str = None
    learning_rate: float = 2e-5
    train_max_seq_length: int = 256
    eval_max_seq_length: int = 256
    predict_max_seq_length: int = 256
    per_gpu_train_batch_size: int = 16
    per_gpu_eval_batch_size: int = 16
    per_gpu_predict_batch_size: int = 16
    seg_len: int = 254
    seg_backoff: int = 64
    num_train_epochs: int = 5
    fold: int = 0
    num_augments: int = 0
    enable_kd: bool = False
    kd_coeff: float = 1.0
    kd_decay: float = 0.995
    enable_sda: bool = False
    sda_teachers: int = 3
    sda_stategy: str = "recent_models"
    sda_empty_first: bool = False
    sda_coeff: float = 1.0
    sda_decay: float = 0.995
    loss_type: str = "CrossEntropyLoss"
    focalloss_gamma: float = 1.5
    focalloss_alpha: List = None
    diceloss_weight: List = None
    model_type: str = "bert"
    model_path: str = None
    train_rate: float = 0.9
    train_sample_rate: float = 1.0
    fp16: bool = True
    seed: int = 8864
    best_index: str = "f1"
    random_type: str = None
    is_english: bool = False
    allow_overlap: bool = False
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_train_examples: int = 0
    confidence: float = 0.35
    enable_nested_entities: bool = False
    emotion_words_file: str = None
    cc: str = None
    brat_data_dir: str = None
    aug_train_only: bool = False

    def __post_init__(self):
        tracking_uri = self.tracking_uri
        if 'TRACKING_URI' in os.environ:
            tracking_uri = os.environ['TRACKING_URI']
        self.tracking_uri = tracking_uri

        artifact_path = self.artifact_path
        if 'ARTIFACT_PATH' in os.environ:
            artifact_path = os.environ['ARTIFACT_PATH']
        self.artifact_path = artifact_path


@dataclass
class NerParams(Params):
    ner_labels: List[str] = field(default_factory=list)
    ner_type: str = "span"
    no_crf_loss: bool = False
    soft_label: bool = False
    ignore_categories: List[str] = None


@dataclass
class SpoParams(Params):
    predicate_labels: List[str] = field(default_factory=list)


@dataclass
class GlueParams(Params):
    glue_labels: List[str] = field(default_factory=list)


@dataclass
class NerAppParams(Params):
    common_params: CommonParams = field(default_factory=CommonParams)
    ner_params: NerParams = field(default_factory=NerParams)


@dataclass
class SpoAppParams(Params):
    common_params: CommonParams = field(default_factory=CommonParams)
    spo_params: SpoParams = field(default_factory=SpoParams)


@dataclass
class GlueAppParams(Params):
    common_params: CommonParams = field(default_factory=CommonParams)
    glue_params: GlueParams = field(default_factory=GlueParams)


def log_global_params(args, experiment_params):
    pass
    #  if args.do_experiment:
    #      run_id = mlflow.active_run().info.run_id
    #      mlflow.log_param("run_id", run_id)
    #      mlflow.log_param("local_id", args.local_id)
    #      mlflow.log_param("args", args)
    #      mlflow.log_param("latest_dir", args.latest_dir)
    #      mlflow.log_param("local_dir", args.local_dir)
    #
    #      mlflow.log_param("experiment_params", experiment_params)
    #      experiment_params.log()


def archive_local_model(args, submission_file=None):
    #  if args.do_experiment:
    #      if submission_file:
    #          mlflow.log_param(f"{args.dataset_name}_submission_file",
    #                           submission_file)
    #          if os.path.exists(submission_file):
    #              mlflow.log_artifact(submission_file)
    #          logger.info(f"Log {submission_file} to tracking.mlflow.")

    if os.path.exists(args.local_dir):
        shutil.rmtree(args.local_dir)
    shutil.copytree(args.latest_dir, args.local_dir)
    logger.info(
        f"Archive local model({args.local_id}) {args.latest_dir} to {args.local_dir}"
    )

    #  os.remove(args.local_id_file)
    #  logger.info(
    #      f"Unlink {args.local_id_file} after archived local model({args.local_id})."
    #  )


#  def augement_entities(all_text_entities, labels_map):
#      aug_tokens = []
#      for i, (guid, text, entities) in enumerate(
#              tqdm(all_text_entities, desc=f"Augement {num_augements}X")):
#
#          #  print(f"-------------------{json_file}--------------------")
#          #  print(text)
#          #  print(entities)
#          #  for entity in entities:
#          #      s = entity['start_pos']
#          #      e = entity['end_pos']
#          #      print(f"{entity['label_type']}: {text[s:e]}")
#          #  print("----------------------------------------")
#          if entities:
#              for ai in range(num_augements):
#                  e_idx = random.randint(0, len(entities) - 1)
#                  entity = entities[e_idx]
#
#                  label_type = entity['label_type']
#                  s = entity['start_pos']
#                  e = entity['end_pos']
#
#                  labels = labels_map[label_type]
#                  idx = random.randint(0, len(labels) - 1)
#                  new_entity_text = labels[idx]
#
#                  text = text[:s] + new_entity_text + text[e:]
#
#                  assert len(new_entity_text) >= 0
#                  delta = len(new_entity_text) - (e - s)
#
#                  entity['end_pos'] = entity['start_pos'] + len(new_entity_text)
#                  entity['mention'] = new_entity_text
#
#                  assert text[
#                      entity['start_pos']:entity['end_pos']] == new_entity_text
#
#                  for n, e in enumerate(entities):
#                      if n > e_idx:
#                          e['start_pos'] += delta
#                          e['end_pos'] += delta
#
#                  aug_tokens.append(
#                      (f"{guid}-a{ai}", text, copy.deepcopy(entities)))
#
#      return aug_tokens


def tensor_to_numpy(t):
    return t.detach().cpu().numpy()


def tensor_to_list(t):
    return t.detach().cpu().numpy().to_list()


def save_args(args, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    #  torch.save(args, os.path.join(model_path, "training_args.bin"))
    logger.info(f"Save args in {model_path}/training_args.json")
    json.dump(
        {
            k: v
            for k, v in args.__dict__.items() if v is None
            or type(v) in [bool, str, int, float, dict, list, tuple]
        },
        open(os.path.join(model_path, "training_args.json"), 'w'),
        ensure_ascii=False,
        indent=2)
