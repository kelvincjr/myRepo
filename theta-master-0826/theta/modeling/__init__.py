#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time
from pathlib import Path
from loguru import logger
from .trainer import generate_dataloader
from .onnx import export_onnx, inference_from_onnx
from .ner_utils import LabeledText, show_ner_datainfo, get_ner_preds_reviews, save_ner_preds, load_ner_examples, load_ner_labeled_examples
from .ner_utils import to_train_poplar, to_reviews_poplar, to_sampling_poplar,  to_poplar
from .glue_utils import show_glue_datainfo, load_glue_examples, save_glue_preds
from .common_args import add_common_args

from dataclasses import dataclass, field
from typing import List
import mlflow


class Params:
    def log(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Params):
                v.log()
            else:
                mlflow.log_param(k, v)

    def debug(self):
        for k, v in self.__dict__.items():
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
    num_augements: int = 0
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
    fp16: bool = True
    seed: int = 8864
    best_index: str = "f1"
    random_type: str = None
    is_english: bool = False

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
    ner_type: str = "crf"
    no_crf_loss: bool = False
    soft_label: bool = False


@dataclass
class GlueParams(Params):
    glue_labels: List[str] = field(default_factory=list)


@dataclass
class NerAppParams(Params):
    common_params: CommonParams = field(default_factory=CommonParams)
    ner_params: NerParams = field(default_factory=NerParams)


@dataclass
class GlueAppParams(Params):
    common_params: CommonParams = field(default_factory=CommonParams)
    glue_params: GlueParams = field(default_factory=GlueParams)


def log_global_params(args, experiment_params):
    if args.do_experiment:
        run_id = mlflow.active_run().info.run_id
        mlflow.log_param("run_id", run_id)
        mlflow.log_param("local_id", args.local_id)
        mlflow.log_param("args", args)
        mlflow.log_param("latest_dir", args.latest_dir)
        mlflow.log_param("local_dir", args.local_dir)

        mlflow.log_param("experiment_params", experiment_params)
        experiment_params.log()


def archive_local_model(args, submission_file):
    if args.do_experiment:
        mlflow.log_param(f"{args.dataset_name}_submission_file",
                         submission_file)
        mlflow.log_artifact(submission_file)
        logger.info(f"Log {submission_file} to tracking.mlflow.")

    import shutil
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


def augement_entities(all_text_entities, labels_map):
    aug_tokens = []
    for i, (guid, text, entities) in enumerate(
            tqdm(all_text_entities, desc=f"Augement {num_augements}X")):

        #  print(f"-------------------{json_file}--------------------")
        #  print(text)
        #  print(entities)
        #  for entity in entities:
        #      s = entity['start_pos']
        #      e = entity['end_pos']
        #      print(f"{entity['label_type']}: {text[s:e]}")
        #  print("----------------------------------------")
        if entities:
            for ai in range(num_augements):
                e_idx = random.randint(0, len(entities) - 1)
                entity = entities[e_idx]

                label_type = entity['label_type']
                s = entity['start_pos']
                e = entity['end_pos']

                labels = labels_map[label_type]
                idx = random.randint(0, len(labels) - 1)
                new_entity_text = labels[idx]

                text = text[:s] + new_entity_text + text[e:]

                assert len(new_entity_text) >= 0
                delta = len(new_entity_text) - (e - s)

                entity['end_pos'] = entity['start_pos'] + len(new_entity_text)
                entity['mention'] = new_entity_text

                assert text[
                    entity['start_pos']:entity['end_pos']] == new_entity_text

                for n, e in enumerate(entities):
                    if n > e_idx:
                        e['start_pos'] += delta
                        e['end_pos'] += delta

                aug_tokens.append(
                    (f"{guid}-a{ai}", text, copy.deepcopy(entities)))

    #  for guid, text, entities in aug_tokens:
    #      text_a = text
    #      for entity in entities:
    #          logger.debug(f"{guid}: text_a: {text_a}")
    #          logger.debug(
    #              f"text_a[entity['start_pos']:entity['end_pos']]: {text_a[entity['start_pos']:entity['end_pos']]}"
    #          )
    #          logger.debug(
    #              f"mention {entity['mention']} in {text_a.find(entity['mention'])}"
    #          )
    #          logger.debug(f"entity: {entity}")
    #          assert text_a[entity['start_pos']:entity['end_pos']] == entity[
    #              'mention']

    return aug_tokens


#  def data_seg_generator(lines, ner_labels, seg_len=0, seg_backoff=0):
#      all_text_entities = []
#      labels_map = {}
#
#      for i, s in enumerate(tqdm(lines)):
#          guid = str(i)
#          text = s['originalText'].strip()
#          entities = s['entities']
#
#          new_entities = []
#          used_span = []
#          entities = sorted(entities, key=lambda e: e['start_pos'])
#          for entity in entities:
#              if entity['label_type'] not in ner_labels:
#                  continue
#              entity['mention'] = text[entity['start_pos']:entity['end_pos']]
#              s = entity['start_pos']
#              e = entity['end_pos']
#
#              overlap = False
#              for us in used_span:
#                  if s >= us[0] and s < us[1]:
#                      overlap = True
#                      break
#                  if e > us[0] and e <= us[1]:
#                      overlap = True
#                      break
#              if overlap:
#                  logger.warning(
#                      f"Overlap! {i} mention: {entity['mention']}, used_span: {used_span}"
#                  )
#                  continue
#              used_span.append((s, e))
#
#              new_entities.append(entity)
#          entities = new_entities
#
#          guid = str(i)
#
#          seg_offset = 0
#          if seg_len <= 0:
#              seg_len = max_seq_length
#
#          for (seg_text, ) in seg_generator((text, ), seg_len, seg_backoff):
#              text_a = seg_text
#
#              seg_start = seg_offset
#              seg_end = seg_offset + min(seg_len, len(seg_text))
#              labels = [
#                  (x['label_type'], x['start_pos'] - seg_offset,
#                   x['end_pos'] - 1 - seg_offset) for x in entities
#                  if x['start_pos'] >= seg_offset and x['end_pos'] <= seg_end
#              ]
#
#              # 没有标注存在的文本片断不用于训练
#              if labels:
#                  yield guid, text_a, None, labels
#
#                  if num_augements > 0:
#                      seg_entities = [{
#                          'start_pos': x['start_pos'] - seg_offset,
#                          'end_pos': x['end_pos'] - seg_offset,
#                          'label_type': x['label_type'],
#                          'mention': x['mention']
#                      } for x in entities if x['start_pos'] >= seg_offset
#                                      and x['end_pos'] <= seg_end]
#                      all_text_entities.append((guid, text_a, seg_entities))
#
#                      for entity in seg_entities:
#                          label_type = entity['label_type']
#                          s = entity['start_pos']  # - seg_offset
#                          e = entity['end_pos']  #- seg_offset
#                          #  print(s, e)
#                          assert e >= s
#                          #  logger.debug(
#                          #      f"seg_start: {seg_start}, seg_end: {seg_end}, seg_offset: {seg_offset}"
#                          #  )
#                          #  logger.debug(f"s: {s}, e: {e}")
#                          assert s >= 0 and e <= len(seg_text)
#                          #  if s >= len(seg_text) or e >= len(seg_text):
#                          #      continue
#
#                          entity_text = seg_text[s:e]
#                          #  print(label_type, entity_text)
#
#                          assert len(entity_text) > 0
#                          if label_type not in labels_map:
#                              labels_map[label_type] = []
#                          labels_map[label_type].append(entity_text)
#
#              seg_offset += seg_len - seg_backoff
#
#      if num_augements > 0:
#          aug_tokens = augement_entities(all_text_entities, labels_map)
#          for guid, text, entities in aug_tokens:
#              text_a = text
#              for entity in entities:
#                  #  logger.debug(f"text_a: {text_a}")
#                  #  logger.debug(
#                  #      f"text_a[entity['start_pos']:entity['end_pos']]: {text_a[entity['start_pos']:entity['end_pos']]}"
#                  #  )
#                  #  logger.debug(
#                  #      f"mention {entity['mention']} in {text_a.find(entity['mention'])}"
#                  #  )
#                  #  logger.debug(f"entity: {entity}")
#                  assert text_a[entity['start_pos']:entity['end_pos']] == entity[
#                      'mention']
#              labels = [
#                  (entity['label_type'], entity['start_pos'],
#                   entity['end_pos'] - 1) for entity in entities
#                  if entity['end_pos'] <= (
#                      min(len(text_a), seg_len) if seg_len > 0 else len(text_a))
#              ]
#              yield guid, text_a, None, labels
