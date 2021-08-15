#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from transformers import (AutoConfig, AutoModelForTokenClassification,
                          BertConfig, BertTokenizer, BertTokenizerFast)
#  from transformers.modeling_bert import BertModel, BertPreTrainedModel
from transformers.models.bert import BertModel, BertPreTrainedModel

from ...losses import DiceLoss, FocalLoss, LabelSmoothingCrossEntropy
from ...utils.multiprocesses import (barrier_leader_process,
                                     barrier_member_processes,
                                     is_multi_processes)
from ..models.linears import PoolerEndLogits, PoolerStartLogits
from ..trainer import Trainer, get_default_optimizer_parameters
from .utils import CNerTokenizer, SeqEntityScore, get_entities

#  import mlflow


class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = config.soft_label
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.focalloss_gamma = config.focalloss_gamma
        self.focalloss_alpha = config.focalloss_alpha
        self.diceloss_weight = config.diceloss_weight

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels,
                                          self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1,
                                          self.num_labels)
        self.init_weights()


#  class BertSpanForNer:
#      def __init__(self, args):
#          self.args = args
#          self.soft_label = args.soft_label
#          self.num_labels = args.num_labels
#          self.loss_type = args.loss_type
#          self.focalloss_gamma = args.focalloss_gamma
#          self.focalloss_alpha = args.focalloss_alpha
#
#          config_class = AutoConfig
#          model_class = AutoConfig
#          config = config_class.from_pretrained(
#              args.model_path,
#              num_labels=args.num_labels,
#              label2id = args.label2id,
#              id2label = args.id2label,
#              cache_dir=args.cache_dir if args.cache_dir else None,
#          )
#          logger.info(f"model_path: {args.model_path}")
#          logger.info(f"config:{config}")
#          self.bert = model_class.from_pretrained(
#              args.model_path,
#              from_tf=bool(".ckpt" in args.model_path),
#              config=config,
#              cache_dir=args.cache_dir if args.cache_dir else None,
#          )
#
#          self.dropout = nn.Dropout(config.hidden_dropout_prob)
#          self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
#          if self.soft_label:
#              self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels,
#                                            self.num_labels)
#          else:
#              self.end_fc = PoolerEndLogits(config.hidden_size + 1,
#                                            self.num_labels)
#          self.init_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                start_positions=None,
                end_positions=None):
        #  subjects_ids=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)

        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len,
                                                 self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits,
                                            -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        avg_logits = (start_logits + end_logits) / 2
        outputs = (
            avg_logits,
            start_logits,
            end_logits,
        ) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            assert self.loss_type in [
                'LabelSmoothingCrossEntropy', 'FocalLoss', 'CrossEntropyLoss'
            ]
            if self.loss_type == 'LabelSmoothingCrossEntropy':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'FocalLoss':
                loss_fct = FocalLoss(gamma=self.focalloss_gamma,
                                     alpha=self.focalloss_alpha)
            elif self.loss_type == 'DiceLoss':
                loss_fct = DiceLoss(weight=self.diceloss_weight)
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss, ) + outputs
            return outputs
        else:
            #  return (0.0, ) + outputs
            #  logger.warning(
            #      f"start_positions: {start_positions}, end_positions: {end_positions}"
            #  )
            return (torch.tensor(0.0).cuda(), ) + outputs
            #  return outputs


class SpanEntityScore(object):
    def __init__(self, id2label, ignore_categories=None):
        self.id2label = id2label
        self.ignore_categories = ignore_categories
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision *
                                                 recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter(
            [f"{x[0]}:{self.id2label[x[0]]}" for x in self.origins])
        found_counter = Counter(
            [f"{x[0]}:{self.id2label[x[0]]}" for x in self.founds])
        right_counter = Counter(
            [f"{x[0]}:{self.id2label[x[0]]}" for x in self.rights])

        total_origin = 0
        total_found = 0
        total_right = 0
        for type_, count in origin_counter.items():
            category = type_.split(':')[1]
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {
                "acc": round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'right': right,
                'found': found,
                'origin': origin
            }
            if self.ignore_categories and category in self.ignore_categories:
                pass
            else:
                total_origin += origin
                total_found += found
                total_right += right
        if self.ignore_categories:
            origin = total_origin
            found = total_found
            right = total_right
        else:
            origin = len(self.origins)
            found = len(self.founds)
            right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {
            'acc': precision,
            'recall': recall,
            'f1': f1,
            'right': right,
            'found': found,
            'origin': origin
        }, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([
            pre_entity for pre_entity in pred_subject
            if pre_entity in true_subject
        ])


MODEL_CLASSES = {
    'bert': (BertConfig, BertSpanForNer, CNerTokenizer),
    #  'bert': (BertConfig, BertSpanForNer, BertTokenizer),
    #  'bert': (BertConfig, BertSpanForNer, BertTokenizerFast),
}


def load_pretrained_tokenizer(args):
    #  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    #  tokenizer = tokenizer_class.from_pretrained(
    #      args.model_path,
    #      do_lower_case=args.do_lower_case,
    #      is_english=args.is_english,
    #      cache_dir=args.cache_dir if args.cache_dir else None,
    #  )
    from ..token_utils import HFTokenizer
    tokenizer = HFTokenizer(os.path.join(args.model_path, 'vocab.txt'),
                            lowercase=args.do_lower_case,
                            cc=args.cc)

    return tokenizer


def load_pretrained_model(args):
    # make sure only the first process in distributed training
    # will download model & vocab
    barrier_member_processes(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.model_path,
        num_labels=args.num_labels,
        loss_type=args.loss_type,
        cache_dir=args.cache_dir if args.cache_dir else None,
        soft_label=args.soft_label,
    )
    setattr(config, 'label2id', args.label2id)
    setattr(config, 'id2label', args.id2label)
    setattr(config, 'soft_label', args.soft_label)
    setattr(config, 'loss_type', args.loss_type)
    setattr(config, 'focalloss_gamma', args.focalloss_gamma)
    setattr(config, 'focalloss_alpha', args.focalloss_alpha)
    setattr(config, 'diceloss_weight', args.diceloss_weight)
    logger.info(f"model_path: {args.model_path}")
    logger.info(f"config:{config}")
    model = model_class.from_pretrained(
        args.model_path,
        from_tf=bool(".ckpt" in args.model_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # make sure only the first process in distributed training
    # will download model & vocab
    barrier_leader_process(args)

    return model


#  def batch_to_input_data(batch):
#      all_input_ids = torch.stack([e.input_ids for e in batch])
#      all_input_mask = torch.stack([e.input_mask for e in batch])
#      all_segment_ids = torch.stack([e.segment_ids for e in batch])
#      all_input_lens = torch.stack([e.input_len for e in batch])
#      all_token_offsets = torch.stack([e.token_offsets for e in batch])
#      all_start_ids = torch.stack([e.start_ids for e in batch])
#      all_end_ids = torch.stack([e.end_ids for e in batch])
#      all_subjects = [e.subjects for e in batch]
#
#      return (all_input_ids, all_input_mask, all_segment_ids, all_start_ids,
#              all_end_ids, all_input_lens, all_subjects, all_token_offsets)


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    #  all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_input_lens = map(
    #      torch.stack, zip(*batch))
    #  from .dataset import batch_to_input_data
    #  all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_input_lens, all_subjects, all_token_offsets = batch_to_input_data(
    #      batch)
    all_input_ids = torch.stack([e.input_ids for e in batch])
    all_attention_mask = torch.stack([e.attention_mask for e in batch])
    all_token_type_ids = torch.stack([e.token_type_ids for e in batch])
    all_input_lens = torch.stack([e.input_len for e in batch])
    all_token_offsets = torch.stack([e.token_offsets for e in batch])

    all_start_ids = torch.stack([e.start_ids for e in batch])
    all_end_ids = torch.stack([e.end_ids for e in batch])
    all_subjects = [e.subjects for e in batch]

    max_len = max(all_input_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_token_offsets = all_token_offsets[:, :max_len]

    all_start_ids = all_start_ids[:, :max_len]
    all_end_ids = all_end_ids[:, :max_len]

    return all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets, all_start_ids, all_end_ids, all_subjects


def bert_extract_item(start_logits,
                      end_logits,
                      lens,
                      confidence,
                      overlap=False):
    num_tokens = lens[0]
    S = []

    starts = torch.argmax(start_logits, -1).cpu().numpy()[0]  #[1:-1]
    ends = torch.argmax(end_logits, -1).cpu().numpy()[0]  #[1:-1]

    # start_logits.shape: (1, sentence_length, num_labels)
    #  logger.debug(f"start_logits: {start_logits.shape}")
    #  start_max_probs = [
    #      f"{start_logits[0][i][x]:.4f}" for i, x in enumerate(starts)
    #  ]
    #  logger.debug(f"start_max_probs: {start_max_probs}")
    #  logger.debug(f"starts: {starts.shape} {starts}")
    starts = [
        x if start_logits[0][i][x] >= confidence else 0
        for i, x in enumerate(starts)
    ]
    ends = [
        x if end_logits[0][i][x] >= confidence else 0
        for i, x in enumerate(ends)
    ]

    #  starts = [starts[0]] + [ x if x != starts[i] else 0 for i, x in enumerate(starts[1:])]
    #  ends = [ends[0]] + [ x if x != ends[i] else 0 for i, x in enumerate(ends[1:])]
    def filter_process(starts):
        new_starts = []
        for i, x in enumerate(starts):
            is_dup = False
            if i < len(starts) - 1 and x == starts[i + 1]:
                is_dup = True
            elif i > 0 and starts[i - 1] == x:
                is_dup = True
            if is_dup:
                new_starts.append(0)
            else:
                new_starts.append(x)
        return new_starts

    #  starts = filter_process(starts)
    #  ends = filter_process(ends)

    #  starts = np.array([x for x in starts if x >= 0 and x < num_tokens])
    #  ends = np.array([x for x in ends if x >= 0 and x < num_tokens])
    starts = starts[:num_tokens]
    ends = ends[:num_tokens]

    #  logger.info(f"start_pred: {starts}")
    #  logger.info(f"end_pred: {ends}")
    #  for i, s_l in enumerate(starts):
    #      if s_l == 0:
    #          continue
    #      for j, e_l in enumerate(ends[i:]):
    #          if s_l == e_l:
    #              S.append((s_l, i, i + j))
    #              break
    #          if i + j < len(starts) - 1 and starts[i + j + 1] != 0:
    #              break
    #  for i in range(len(starts) - 1):
    last_j = -1
    for i in range(len(starts)):
        if i <= last_j:
            continue
        s_l = starts[i]
        if s_l == 0:
            continue
        for j, e_l in enumerate(ends[i:]):
            if s_l == e_l:
                if not overlap:
                    #  if sum(starts[i + 1:i + j + 1]) == 0:
                    S.append((int(s_l), i, i + j))
                    last_j = j
                    i = j + 1
                    break
                else:
                    if sum(starts[i + 1:i + j + 1]) != 0:
                        break
                    S.append((int(s_l), i, i + j))
            if i + j < len(starts) - 1 and starts[i + j + 1] != 0:
                break
    #  S = [x for x in S if x[1] <= x[2]]

    #  for x in S:
    #      assert x[1] >= 0 and x[2] >= 0 and x[1] <= x[2], f"S: {S}"

    return S


def load_model(args):
    model = load_pretrained_model(args)
    model.to(args.device)
    return model


def build_default_model(args):
    """
    自定义模型
    规格要求返回模型(model)、优化器(optimizer)、调度器(scheduler)三元组。
    """

    # -------- model --------
    model = load_pretrained_model(args)
    model.to(args.device)

    # -------- optimizer --------
    from transformers.optimization import AdamW
    optimizer_parameters = get_default_optimizer_parameters(
        model, args.weight_decay)
    optimizer = AdamW(optimizer_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon,
                      correct_bias=False)

    # -------- scheduler --------
    from transformers.optimization import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.total_steps * args.warmup_rate,
        num_training_steps=args.total_steps)

    return model, optimizer, scheduler


def init_labels(args, labels):

    #  args.ner_labels = ['[unused1]', '[unused2]', '[unused3]'] + labels
    #  args.id2label = {i: label for i, label in enumerate(args.ner_labels)}
    #  args.label2id = {label: i for i, label in enumerate(args.ner_labels)}
    args.ner_labels = ['[unused1]'] + labels
    args.id2label = {i: label for i, label in enumerate(args.ner_labels)}
    args.label2id = {label: i for i, label in enumerate(args.ner_labels)}
    args.num_labels = len(args.label2id)

    #  args.id2label = {i + 100: label for i, label in enumerate(args.ner_labels)}
    #  args.label2id = {label: i + 100 for i, label in enumerate(args.ner_labels)}
    #
    #  args.ner_labels = ['[unused1]'] + labels
    #  args.id2label[0] = '[unused1]'
    #  args.label2id['[unused1]'] = 0
    #  args.num_labels = len(args.label2id)

    logger.info(f"args.label2id: {args.label2id}")
    logger.info(f"args.id2label: {args.id2label}")
    logger.info(f"args.num_labels: {args.num_labels}")


class NerTrainer(Trainer):
    def __init__(self, args, ner_labels, build_model=None, tokenizer=None):
        super(NerTrainer, self).__init__(args)
        init_labels(args, ner_labels)
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = load_pretrained_tokenizer(args)

        if build_model is None:
            self.build_model = build_default_model
        else:
            self.build_model = build_model

        self.label2id = args.label2id
        self.collate_fn = collate_fn

    #  def examples_to_dataset(self, examples, max_seq_length):
    #      from .dataset import examples_to_dataset
    #      return examples_to_dataset(examples, self.label2id, self.tokenizer,
    #                                 max_seq_length)

    def encode_examples(self,
                        examples,
                        max_seq_length,
                        enttype_labels=None,
                        epoch=-1):
        from .dataset import encode_examples
        return encode_examples(examples,
                               self.label2id,
                               self.tokenizer,
                               max_seq_length,
                               enttype_labels=enttype_labels,
                               epoch=epoch)

    def batch_to_inputs(self, args, batch, known_labels=True):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "start_positions": batch[5],
            "end_positions": batch[6],
        }
        if args.model_type != "distilbert":
            # XLM and RoBERTa don"t use segment_ids
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "xlnet"] else None)

        return inputs

    #  def generate_dataloader(self, args, dataset, batch_size, keep_order=True):
    #
    #      Sampler = SequentialSampler if keep_order else RandomSampler
    #      sampler = DistributedSampler(dataset) if is_multi_processes(
    #          args) else Sampler(dataset)
    #      dataloader = DataLoader(dataset,
    #                              sampler=sampler,
    #                              batch_size=batch_size,
    #                              collate_fn=collate_fn)
    #      return dataloader

    def on_eval_start(self, args, eval_dataset):
        self.metric = SpanEntityScore(args.id2label,
                                      ignore_categories=args.ignore_categories)
        self.eval_logs = []

    #  def on_eval_step(self, args, eval_dataset, step, model, inputs, outputs):
    def on_eval_step(self, args, model, step, batch):
        all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets, all_start_ids, all_end_ids, all_subjects = batch

        eval_loss = 0.0
        num_eval_steps = 0
        for i in range(all_input_ids.size()[0]):
            inputs = {
                "input_ids": all_input_ids[i].view(1, -1),
                "attention_mask": all_attention_mask[i].view(1, -1),
                "start_positions": all_start_ids[i].view(1, -1),
                "end_positions": all_end_ids[i].view(1, -1),
            }
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (all_token_type_ids[i].view(
                    1, -1) if args.model_type in ["bert", "xlnet"] else None)

            outputs = model(**inputs)
            tmp_eval_loss, avg_logits, start_logits, end_logits = outputs[:4]
            #  tmp_eval_loss, start_logits, end_logits = outputs[:3]
            eval_loss += tmp_eval_loss
            num_eval_steps += 1

            start_logits = F.softmax(start_logits, -1)
            end_logits = F.softmax(end_logits, -1)
            num_tokens = int(all_input_lens[i])
            R = bert_extract_item(start_logits,
                                  end_logits, [num_tokens],
                                  args.confidence,
                                  overlap=args.allow_overlap)

            T = all_subjects[i]

            #  logger.warning(f"R: {R}")
            #  logger.warning(f"T: {T}")

            self.metric.update(true_subject=T, pred_subject=R)

            def calculate_eval(true_subject, pred_subject):
                origins = true_subject
                founds = pred_subject
                rights = [
                    pre_entity for pre_entity in pred_subject
                    if pre_entity in true_subject
                ]
                origin = len(origins)
                found = len(founds)
                right = len(rights)
                recall = 0 if origin == 0 else (right / origin)
                precision = 0 if found == 0 else (right / found)
                f1 = 0. if recall + precision == 0 else (
                    2 * precision * recall) / (precision + recall)
                return precision, recall, f1

            precision, recall, f1 = calculate_eval(T, R)

            token_offsets = all_token_offsets[i]
            #  label_entities = [[
            #      args.id2label[x[0]], token_offsets[x[1]][0].item(),
            #      token_offsets[x[2]][-1].item() - 1
            #  ] for x in R]
            self.eval_logs.append(
                (num_tokens, R, token_offsets, (precision, recall, f1)))

        eval_loss = eval_loss / num_eval_steps
        eval_info, entity_info = self.metric.result()
        results = {f'{key}': value for key, value in eval_info.items()}
        results['loss'] = eval_loss

        #  if args.do_experiment:
        #      mlflow.log_metric('loss', eval_loss.item())
        #      for key, value in eval_info.items():
        #          mlflow.log_metric(key, value)

        return (eval_loss, ), results

    def on_eval_end(self, args, eval_examples):
        from ...utils.ner_utils import get_ner_results
        results = get_ner_results(self.metric)

        eval_logs_file = f"{args.latest_dir}/eval_logs_{args.epoch}.json"

        def entity_text(example, token_offsets, m, s, e):
            s = token_offsets[s][0].item()
            e = token_offsets[e][-1].item()
            return f"{args.id2label[m]}|({m}, {s}, {e})|{example.text[s:e]}"

        all_eval_logs = [{
            'text':
            example.text,
            'num_tokens':
            num_tokens,
            'subjects': [
                entity_text(example, token_offsets, m, s, e)
                for m, s, e in example.subjects
            ],
            'right': [
                entity_text(example, token_offsets, m, s, e)
                for m, s, e in sorted(list(set(R) & set(example.subjects)),
                                      key=lambda x: x[1])
            ],
            'miss': [
                entity_text(example, token_offsets, m, s, e)
                for m, s, e in sorted(list(set(example.subjects) - set(R)),
                                      key=lambda x: x[1])
            ],
            'extra': [
                entity_text(example, token_offsets, m, s, e)
                for m, s, e in sorted(list(set(R) - set(example.subjects)),
                                      key=lambda x: x[1])
            ],
            'pred':
            [entity_text(example, token_offsets, m, s, e) for m, s, e in R],
            'acc_recall_f1':
            x
        } for example, (
            num_tokens,
            R, token_offsets,
            x) in tqdm(zip(eval_examples, self.eval_logs), desc="eval_logs")]
        json.dump(all_eval_logs,
                  open(eval_logs_file, 'w'),
                  ensure_ascii=False,
                  indent=2)

        logger.warning(f"Saved eval logs in {eval_logs_file}")

        return results

    def on_predict_start(self, args, test_features):
        self.pred_results = []

    def on_predict_step(self, args, model, step, batch):
        all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets, all_start_ids, all_end_ids, all_subjects = batch

        for i in range(all_input_ids.size()[0]):
            inputs = {
                "input_ids": all_input_ids[i].view(1, -1),
                "attention_mask": all_attention_mask[i].view(1, -1),
            }
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (all_token_type_ids[i].view(
                    1, -1) if args.model_type in ["bert", "xlnet"] else None)
            token_offsets = all_token_offsets[i]

            outputs = model(**inputs)
            _, _, start_logits, end_logits = outputs[:4]
            #  _, start_logits, end_logits = outputs[:3]

            start_logits = F.softmax(start_logits, -1)
            end_logits = F.softmax(end_logits, -1)
            num_tokens = int(all_input_lens[i])
            R = bert_extract_item(start_logits,
                                  end_logits, [num_tokens],
                                  args.confidence,
                                  overlap=args.allow_overlap)

            if R:
                label_entities = [[
                    args.id2label[x[0]], token_offsets[x[1]][0].item(),
                    token_offsets[x[2]][-1].item() - 1
                ] for x in R]
                if args.ignore_categories:
                    label_entities = [
                        x for x in label_entities
                        if x[0] not in args.ignore_categories
                    ]
                label_entities = [
                    x for x in label_entities
                    if x[1] <= x[2] and x[1] >= 0 and x[2] >= 0
                ]

            else:
                label_entities = []

            #  logger.debug(f"{label_entities}")
            json_d = {}
            json_d['id'] = step
            #  tag_seq = [args.id2label[x] for x in preds]
            #  json_d['tag_seq'] = " ".join(tag_seq)
            json_d['entities'] = label_entities

            #  logger.debug(f"{json_d}")

            self.pred_results.append(json_d)

    def on_predict_end(self, args, test_dataset):
        return self.pred_results
