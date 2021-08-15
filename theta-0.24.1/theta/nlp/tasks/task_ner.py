#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Type

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm

from transformers import AutoModel, BertModel, BertPreTrainedModel

from ...modules import FocalLoss, LabelSmoothingCrossEntropy
from ..arguments import (DataArguments, ModelArguments, TaskArguments,
                         TrainingArguments,
                         generate_method_kwargs_from_arguments)
from ..data.samples import GlueSamples
#  from ...losses import DiceLoss, FocalLoss
#  from .ner_decodes import crf_decode, mrc_decode, span_decode
from .task import BaseDataset, BaseTask, TaskData, TaskRunner, TransformerModel

#  def softmax(x):
#      e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
#      return e_x / e_x.sum(axis=1).reshape(-1, 1)


def get_p_r_f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])


def generate_char2token(offset_mapping, num_text_len):
    char2token = [-1] * num_text_len

    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            continue
        #  token2char[i] = start
        for j in range(start, end):
            char2token[j] = i

    return char2token


def generate_token2chars(offset_mapping):
    token2char = [-1] * (len(offset_mapping) + 1)

    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            token2char[i] = -1
            continue
        token2char[i] = start
        token2char[i + 1] = end

    return token2char


# ------------------------------ Dataset ------------------------------
class NerDataset(BaseDataset):
    """
    完成模型输入数据的编码工作
    """
    def __init__(self, *args, **kwargs):
        super(NerDataset, self).__init__(*args, **kwargs)

    def _encode_item(self, x):
        guid, text, _, tags = x

        #  logger.info(f"{guid}, {text}, tags: {tags}")

        def seg_encode(text, tags):
            batch_texts = [text]
            batch_tags = [tags]
            # -------- input_ids, attention_mask, token_type_ids --------
            encodings = self.tokenizer.batch_encode_plus(
                batch_texts,
                padding=self.data_args.padding,
                max_length=self.data_args.max_length,
                add_special_tokens=True,
                truncation=True,
                return_offsets_mapping=True)

            batch_input_ids = torch.from_numpy(
                np.array(encodings.input_ids, dtype=np.int64))
            batch_attention_mask = torch.from_numpy(
                np.array(encodings.attention_mask, dtype=np.int64))
            batch_token_type_ids = torch.from_numpy(
                np.array(encodings.token_type_ids, dtype=np.int64))

            batch_offset_mapping = encodings.offset_mapping
            batch_tokens = [
                #
                [text[b:e] for b, e in offset_mapping if b > 0 or e > 0]
                #
                for text, offset_mapping in zip(batch_texts,
                                                batch_offset_mapping)
            ]
            batch_char2tokens = [
                generate_char2token(offset_mapping,
                                    len(text)) for offset_mapping, text in zip(
                                        batch_offset_mapping, batch_texts)
            ]
            batch_token2chars = [
                generate_token2chars(offset_mapping)
                for offset_mapping in batch_offset_mapping
            ]

            #  for text, offset_mapping in zip(batch_texts, batch_offset_mapping):
            #      tokens = [
            #          text[b:e] for b, e in offset_mapping if b > 0 or e > 0
            #      ]

            input_ids = batch_input_ids[0]
            attention_mask = batch_attention_mask[0]
            token_type_ids = batch_token_type_ids[0]
            tokens = batch_tokens[0]
            char2token = batch_char2tokens[0]
            token2char = batch_token2chars[0]

            # -------- labels --------
            if tags is not None:
                num_tokens = input_ids.shape[0]
                start_ids = [0] * num_tokens
                end_ids = [0] * num_tokens
                for tag in tags:
                    c = tag['category']
                    s = tag['start']
                    m = tag['mention']
                    e = s + len(m) - 1

                    #  logger.info(f"text: {text}")
                    #  logger.info(f"tokens: {tokens}")
                    #  logger.info(f"tag: {tag}")
                    #  logger.info(f"s: {s}, e: {e}, char2token: {char2token}")
                    s = char2token[s]
                    if s < 1:
                        logger.warning(f"{guid}, {text}, tags: {tags}")
                    assert s >= 1
                    e = char2token[e]
                    if e < 1:
                        logger.warning(f"{guid}, {text}, tags: {tags}")
                    assert e >= 1
                    #  logger.debug(
                    #      f"num_tokens: {num_tokens}, s: {s}, e: {e}, c: {c}, m: {m}, text: {text}"
                    #  )
                    c_id = self.label2id[c]
                    #  assert c_id < 10, f"c_id: {c_id}"
                    #  logger.info(f"c_id: {c_id}")
                    start_ids[s] = c_id
                    end_ids[e] = c_id

                start_ids = torch.from_numpy(
                    np.array([start_ids], dtype=np.int64))[0]
                end_ids = torch.from_numpy(np.array([end_ids],
                                                    dtype=np.int64))[0]
            else:
                start_ids = None
                end_ids = None

            return {
                'tokens': tokens,
                'char2token': char2token,
                'token2char': token2char,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'start_ids': start_ids,
                'end_ids': end_ids,
                'tags': tags
            }

        all_encoded = []
        seg_len = self.data_args.max_length - 2
        seg_stride = int(seg_len / 2)
        seg_offset = 0
        while seg_offset < len(text):
            s_seg = seg_offset
            e_seg = seg_offset + seg_len - 1
            seg_text = text[s_seg:e_seg + 1]
            if len(seg_text) == 0:
                continue
            e_seg = min(e_seg, len(seg_text) - 1)
            self.seg_spans.append((guid, s_seg, e_seg))

            if tags is not None:
                seg_tags = []
                if tags:
                    for tag in tags:
                        s = tag['start']
                        e = s + len(tag['mention']) - 1
                        if s >= s_seg and e <= e_seg:
                            seg_tag = deepcopy(tag)
                            seg_tag['start'] -= seg_offset
                            seg_tags.append(seg_tag)
                            #  logger.info(
                            #      f"text: {text}\n"
                            #      f"tag: {tag}\n"
                            #      f"seg_len: {seg_len}, seg_stride: {seg_stride}\n"
                            #      f"seg_offset: {seg_offset}\n"
                            #      f"seg_tag: {seg_tag}\n")
                            assert (
                                seg_text[s:e + 1] == tag['mention'],
                                f"text: {text}\n"
                                f"tag: {tag}\n"
                                f"seg_len: {seg_len}, seg_stride: {seg_stride}\n"
                                f"seg_offset: {seg_offset}\n"
                                f"seg_tag: {seg_tag}\n")

                if seg_tags:
                    encoded = seg_encode(seg_text, seg_tags)
                    all_encoded.append(encoded)
                else:
                    if self.data_args.preserve_no_entity:
                        encoded = seg_encode(seg_text, [])
                        all_encoded.append(encoded)
            else:
                encoded = seg_encode(seg_text, None)
                all_encoded.append(encoded)

            seg_offset += seg_stride

        return all_encoded

    @classmethod
    def collate_fn(cls, batch):
        stacked_batch = {}
        #  for key in [
        #          'input_ids', 'attention_mask', 'token_type_ids', 'start_ids',
        #          'end_ids', 'tags'
        #  ]:
        #      key_batch = [e[key] for e in batch if e[key] is not None]
        #      if key_batch:
        #          if key == 'tags':
        #              stacked_batch['tags'] = key_batch
        #          else:
        #              #  if key == 'start_ids':
        #              #      logger.warning(f"key: {key}, key_batch: {key_batch}")
        #              batch_values = torch.stack(key_batch)
        #              stacked_batch[key] = batch_values
        #      else:
        #          stacked_batch[key] = None

        not_none_tensor_keys = [
            'input_ids', 'attention_mask', 'token_type_ids'
        ]
        maybe_none_tensor_keys = ['start_ids', 'end_ids']
        not_tensor_keys = ['tokens', 'char2token', 'token2char', 'tags']

        # not None tensors
        for key in not_none_tensor_keys:
            key_batch = [e[key] for e in batch if e[key] is not None]
            batch_values = torch.stack(key_batch)
            stacked_batch[key] = batch_values
        # maybe None tensors
        for key in maybe_none_tensor_keys:
            key_batch = [e[key] for e in batch if e[key] is not None]
            if key_batch:
                batch_values = torch.stack(key_batch)
                stacked_batch[key] = batch_values
            else:
                stacked_batch[key] = None
        # not tensors
        for key in not_tensor_keys:
            key_batch = [e[key] for e in batch if e[key] is not None]
            stacked_batch[key] = key_batch

        return stacked_batch


# ------------------------------ TaskData ------------------------------
class NerData(TaskData):
    """
    指定任务专属的Dataset类型，并提供训练、验证、测试集实例。
    """
    def __init__(self, *args, **kwargs):
        super(NerData, self).__init__(*args, **kwargs)

    def build_train_dataset(self):
        return NerDataset(self.data_args, self._splitted_train_samples.rows,
                          self.label2id, self.tokenizer)

    def build_val_dataset(self):
        return NerDataset(self.data_args, self._splitted_val_samples.rows,
                          self.label2id, self.tokenizer)

    def build_test_dataset(self):
        return NerDataset(self.data_args, self.test_samples.rows,
                          self.label2id, self.tokenizer)


# ------------------------------ SpanModel ------------------------------
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)),
                           p=self.dropout_rate,
                           training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_ids=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_ids], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x


class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = False  #config.soft_label
        self.num_labels = config.num_labels
        #  self.loss_type = "CrossEntropyLoss"  #"FocalLoss"  #config.loss_type
        self.loss_type = "FocalLoss"
        self.focalloss_gamma = 2.0  #config.focalloss_gamma
        self.focalloss_alpha = 0.25  #config.focalloss_alpha
        #  self.diceloss_weight = config.diceloss_weight
        logger.warning(f"BertSpanForNer.num_labels: {self.num_labels}")

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

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                start_ids=None,
                end_ids=None,
                **kwargs):
        #  subjects_ids=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)

        if start_ids is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len,
                                                 self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_ids.unsqueeze(2), 1)
            else:
                label_logits = start_ids.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits,
                                            -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        #  avg_logits = (start_logits + end_logits) / 2
        #  outputs = (
        #      #  avg_logits,
        #      start_logits,
        #      end_logits,
        #  ) + outputs[2:]
        outputs = (start_logits, end_logits)

        if start_ids is not None and end_ids is not None:
            assert self.loss_type in [
                'LabelSmoothingCrossEntropy', 'FocalLoss', 'CrossEntropyLoss'
            ]
            if self.loss_type == 'LabelSmoothingCrossEntropy':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'FocalLoss':
                loss_fct = FocalLoss(gamma=self.focalloss_gamma)
                #  alpha=self.focalloss_alpha)
            elif self.loss_type == 'DiceLoss':
                loss_fct = DiceLoss(weight=self.diceloss_weight)
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_ids.view(-1)[active_loss]
            active_end_labels = end_ids.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss, ) + outputs
            return outputs
        else:
            #  return (0.0, ) + outputs
            #  logger.warning(
            #      f"start_ids: {start_ids}, end_ids: {end_ids}"
            #  )
            #  return (torch.tensor(0.0).cuda(), ) + outputs
            return outputs

            #  return outputs


class BertSpanModel(TransformerModel):
    def __init__(
        self,
        model_name_or_path,
        num_labels,
        tokenizer=None,
        dropout_prob=0.1,
        #  loss_type='CrossEntropyLoss',
        loss_type='FocalLoss',
        #  loss_type='LabelSmoothingCrossEntropy',
        #  loss_type='DiceLoss',
        **kwargs):

        # for TransformerModel.load_from_config()
        self.num_labels = num_labels
        super(BertSpanModel, self).__init__(model_name_or_path,
                                            tokenizer=tokenizer,
                                            automodel_cls=BertSpanForNer)
        self.init_weights()

    def load_from_pretrained(self, model_path):
        self.transformer = BertSpanForNer.from_pretrained(
            model_path, num_labels=self.num_labels)

    def load_from_config(self):
        #  self.transformer = BertSpanForNer.from_config(self.config)
        setattr(self.config, 'num_labels', self.num_labels)
        self.transformer = BertSpanForNer(self.config)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                start_ids=None,
                end_ids=None,
                pseudo=None,
                tags=None):

        outputs = self.transformer(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   start_ids=start_ids,
                                   end_ids=end_ids,
                                   tags=tags)
        if tags is not None:
            loss, start_logits, end_logits = outputs
            return loss, start_logits, end_logits
        else:
            start_logits, end_logits = outputs
            return start_logits, end_logits

    @classmethod
    def decode_ents_0_23_0(start_logits,
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

    @classmethod
    def decode_ents(cls, start_probs, end_probs, batch_lens, confidence=0.0):
        """
        start_probs.shape: (batch_size, max_length, num_labels)
        end_probs.shape: (batch_size, max_length, num_labels)
        """
        #  logger.info(f"start_probs.shape: {start_probs.shape}")
        #  logger.info(f"end_probs.shape: {end_probs.shape}")

        #  start_probs = start_probs[:, 1:-1]
        #  end_probs = end_probs[:, 1:-1]

        #  logger.info(f"start_probs: {start_probs}")
        #  logger.info(f"end_probs: {end_probs}")
        """
        start_preds.shape: (batch_size, max_length -2)
        end_preds.shape: (batch_size, max_length -2)
        """
        start_preds = np.argmax(start_probs, -1)
        end_preds = np.argmax(end_probs, -1)
        #  logger.info(f"start_preds: {start_preds.shape} {start_preds[:3,:]}")
        #  logger.info(f"end_preds: {end_preds.shape} {end_preds[:3,:]}")

        if confidence > 0.0:
            for i, x in enumerate(start_preds):
                start_preds[i] = [
                    category if start_probs[i][j][category] > confidence else 0
                    for j, category in enumerate(start_preds[i])
                ]
            for i, x in enumerate(end_preds):
                end_preds[i] = [
                    category if end_probs[i][j][category] > confidence else 0
                    for j, category in enumerate(end_preds[i])
                ]

        final_predict_ents = []
        for start_pred, end_pred, text_len in zip(start_preds, end_preds,
                                                  batch_lens):
            start_pred = start_pred[:text_len]
            end_pred = end_pred[:text_len]
            predict_ents = defaultdict(list)
            last_j = -1
            for i, s_type in enumerate(start_pred):
                if s_type == 0:
                    continue
                if i <= last_j:
                    continue
                for j, e_type in enumerate(end_pred[i:]):
                    if s_type == e_type:
                        last_j = j
                        s = i
                        e = j + i
                        predict_ents[s_type].append((s, e))
                        break
                    if i + j < len(start_pred) - 1 and start_pred[i + j +
                                                                  1] != 0:
                        break
            final_predict_ents.append(predict_ents)
        #  logger.warning(f"final_predict_ents: {final_predict_ents}")

        return final_predict_ents


class ___BertSpanModel(TransformerModel):
    def __init__(
        self,
        model_name_or_path,
        num_labels,
        tokenizer=None,
        dropout_prob=0.1,
        #  loss_type='CrossEntropyLoss',
        loss_type='FocalLoss',
        #  loss_type='LabelSmoothingCrossEntropy',
        #  loss_type='DiceLoss',
        **kwargs):
        super(BertSpanModel, self).__init__(model_name_or_path,
                                            tokenizer=tokenizer,
                                            automodel_cls=AutoModel)
        config = self.config
        self.soft_label = False
        self.num_labels = num_labels
        self.loss_type = loss_type
        self.focalloss_gamma = 2.0
        self.focalloss_alpha = 0.25
        #  self.diceloss_weight = config.diceloss_weight

        #  self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels,
                                          self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1,
                                          self.num_labels)
        self.init_weights()

    def load_from_pretrained(self, model_path):
        #      automodel_cls = AutoModel
        #      super(BertSpanModel,
        #            self)._load_from_pretrained(model_path,
        #                                        automodel_cls=automodel_cls)
        self.transformer = AutoModel.from_pretrained(model_path)

    def load_from_config(self):
        self.transformer = AutoModel.from_config(self.config)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                start_ids=None,
                end_ids=None,
                pseudo=None,
                tags=None):
        #  subjects_ids=None):
        outputs = self.transformer(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)

        if start_ids is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len,
                                                 self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_ids.unsqueeze(2), 1)
            else:
                label_logits = start_ids.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits,
                                            -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        avg_logits = (start_logits + end_logits) / 2
        #  outputs = (
        #      avg_logits,
        #      start_logits,
        #      end_logits,
        #  ) + outputs[2:]
        outputs = (start_logits, end_logits)

        if start_ids is not None and end_ids is not None:
            assert self.loss_type in [
                'LabelSmoothingCrossEntropy', 'FocalLoss', 'CrossEntropyLoss',
                'DiceLoss'
            ]
            if self.loss_type == 'LabelSmoothingCrossEntropy':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'FocalLoss':
                loss_fct = FocalLoss(gamma=self.focalloss_gamma)
                #  alpha=self.focalloss_alpha)
            elif self.loss_type == 'DiceLoss':
                loss_fct = DiceLoss()
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_ids.view(-1)[active_loss]
            active_end_labels = end_ids.view(-1)[active_loss]

            # FocalLoss + CrossEntropyLoss
            #  loss_fct_focal = FocalLoss(gamma=self.focalloss_gamma)
            #  loss_fct_ce = CrossEntropyLoss()
            #
            #  start_loss_0 = loss_fct_ce(active_start_logits,
            #                             active_start_labels)
            #  end_loss_0 = loss_fct_ce(active_end_logits, active_end_labels)
            #  start_loss_1 = loss_fct_focal(active_start_logits,
            #                                active_start_labels)
            #  end_loss_1 = loss_fct_focal(active_end_logits, active_end_labels)
            #
            #  start_loss = (start_loss_0 + start_loss_1) / 2
            #  end_loss = (end_loss_0 + end_loss_1) / 2
            #  #
            #  start_loss = loss_fct(active_start_logits, active_start_labels)
            #  end_loss = loss_fct(active_end_logits, active_end_labels)

            #  FocalLoss + CrossEntropyLoss + LabelSmoothingCrossEntropy
            #  loss_fct_focal = FocalLoss(gamma=self.focalloss_gamma)
            #  loss_fct_ce = CrossEntropyLoss()
            #  loss_fct_lsce = LabelSmoothingCrossEntropy()
            #
            #  start_loss_0 = loss_fct_ce(active_start_logits,
            #  active_start_labels)
            #  end_loss_0 = loss_fct_ce(active_end_logits, active_end_labels)
            #
            #  start_loss_1 = loss_fct_focal(active_start_logits,
            #  active_start_labels)
            #  end_loss_1 = loss_fct_focal(active_end_logits, active_end_labels)
            #
            #  start_loss_2 = loss_fct_lsce(active_start_logits,
            #  active_start_labels)
            #  end_loss_2 = loss_fct_lsce(active_end_logits, active_end_labels)
            #
            #  start_loss = (start_loss_0 + start_loss_1 + start_loss_2) / 3
            #  end_loss = (end_loss_0 + end_loss_1 + end_loss_2) / 3

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)

            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss, ) + outputs
            return outputs
        else:
            #  return (0.0, ) + outputs
            #  logger.warning(
            #      f"start_ids: {start_ids}, end_ids: {end_ids}"
            #  )
            #  return (torch.tensor(0.0).cuda(), ) + outputs
            return outputs

    #  @classmethod
    #  def decode_ents(cls, start_logits, end_logits):
    #      """
    #      start_logits.shape: (batch_size, max_length, num_labels)
    #      end_logits.shape: (batch_size, max_length, num_labels)
    #      """
    #      #  logger.info(f"start_logits.shape: {start_logits.shape}")
    #      #  logger.info(f"end_logits.shape: {end_logits.shape}")
    #      start_logits = start_logits[:, 1:-1]
    #      end_logits = end_logits[:, 1:-1]
    #
    #      logger.info(f"start_logits: {start_logits}")
    #      logger.info(f"end_logits: {end_logits}")
    #      """
    #      start_preds.shape: (batch_size, max_length -2)
    #      end_preds.shape: (batch_size, max_length -2)
    #      """
    #      start_preds = np.argmax(start_logits, -1)
    #      end_preds = np.argmax(end_logits, -1)
    #      #  logger.info(f"start_preds: {start_preds.shape} {start_preds[:3,:]}")
    #      #  logger.info(f"end_preds: {end_preds.shape} {end_preds[:3,:]}")
    #
    #      confidence = 0.35
    #      for i, x in enumerate(start_preds):
    #          start_preds[i] = [
    #              category if start_logits[i][j][category] > confidence else 0
    #              for j, category in enumerate(start_preds[i])
    #          ]
    #      for i, x in enumerate(end_preds):
    #          end_preds[i] = [
    #              category if end_logits[i][j][category] > confidence else 0
    #              for j, category in enumerate(end_preds[i])
    #          ]
    #
    #      final_predict_ents = []
    #      for start_pred, end_pred in zip(start_preds, end_preds):
    #          predict_ents = defaultdict(list)
    #          for i, s_type in enumerate(start_pred):
    #              if s_type == 0:
    #                  continue
    #              for j, e_type in enumerate(end_pred[i:]):
    #                  if s_type == e_type:
    #                      s = i
    #                      e = j + i
    #                      predict_ents[s_type].append((s, e))
    #                      break
    #          final_predict_ents.append(predict_ents)
    #      #  logger.warning(f"final_predict_ents: {final_predict_ents}")
    #
    #      return final_predict_ents

    @classmethod
    def decode_ents(cls, start_probs, end_probs):
        """
        start_probs.shape: (batch_size, max_length, num_labels)
        end_probs.shape: (batch_size, max_length, num_labels)
        """
        #  logger.info(f"start_probs.shape: {start_probs.shape}")
        #  logger.info(f"end_probs.shape: {end_probs.shape}")
        start_probs = start_probs[:, 1:-1]
        end_probs = end_probs[:, 1:-1]

        #  logger.info(f"start_probs: {start_probs}")
        #  logger.info(f"end_probs: {end_probs}")
        """
        start_preds.shape: (batch_size, max_length -2)
        end_preds.shape: (batch_size, max_length -2)
        """
        start_preds = np.argmax(start_probs, -1)
        end_preds = np.argmax(end_probs, -1)
        #  logger.info(f"start_preds: {start_preds.shape} {start_preds[:3,:]}")
        #  logger.info(f"end_preds: {end_preds.shape} {end_preds[:3,:]}")

        confidence = 0.5
        for i, x in enumerate(start_preds):
            start_preds[i] = [
                category if start_probs[i][j][category] > confidence else 0
                for j, category in enumerate(start_preds[i])
            ]
        for i, x in enumerate(end_preds):
            end_preds[i] = [
                category if end_probs[i][j][category] > confidence else 0
                for j, category in enumerate(end_preds[i])
            ]

        final_predict_ents = []
        for start_pred, end_pred in zip(start_preds, end_preds):
            predict_ents = defaultdict(list)
            for i, s_type in enumerate(start_pred):
                if s_type == 0:
                    continue
                for j, e_type in enumerate(end_pred[i:]):
                    if s_type == e_type:
                        s = i
                        e = j + i
                        predict_ents[s_type].append((s, e))
                        break
            final_predict_ents.append(predict_ents)
        #  logger.warning(f"final_predict_ents: {final_predict_ents}")

        return final_predict_ents


#  class SpanModel(TransformerModel):
#      def __init__(self,
#                   model_name_or_path,
#                   num_labels,
#                   tokenizer=None,
#                   dropout_prob=0.1,
#                   loss_type='ce',
#                   **kwargs):
#          """
#          tag the subject and object corresponding to the predicate
#          :param loss_type: train loss type in ['ce', 'ls_ce', 'focal']
#          """
#          super(SpanModel, self).__init__(model_name_or_path,
#                                          tokenizer=tokenizer)
#
#          out_dims = self.config.hidden_size
#
#          mid_linear_dims = kwargs.pop('mid_linear_dims', 128)
#
#          self.num_labels = num_labels
#
#          self.mid_linear = nn.Sequential(nn.Linear(out_dims, mid_linear_dims),
#                                          nn.ReLU(), nn.Dropout(dropout_prob))
#
#          out_dims = mid_linear_dims
#
#          self.start_fc = nn.Linear(out_dims, num_labels)
#          self.end_fc = nn.Linear(out_dims, num_labels)
#
#          reduction = 'none'
#          if loss_type == 'ce':
#              self.criterion = nn.CrossEntropyLoss(reduction=reduction)
#          elif loss_type == 'ls_ce':
#              self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
#          else:
#              self.criterion = FocalLoss(reduction=reduction)
#
#          self.loss_weight = nn.Parameter(torch.FloatTensor(1),
#                                          requires_grad=True)
#          self.loss_weight.data.fill_(-0.2)
#
#          init_blocks = [self.mid_linear, self.start_fc, self.end_fc]
#
#          self._init_weights(init_blocks)
#
#      def forward(self,
#                  input_ids,
#                  attention_mask,
#                  token_type_ids,
#                  start_ids=None,
#                  end_ids=None,
#                  pseudo=None,
#                  tags=None):
#
#          #  logger.warning(
#          #      f"input_ids.shape: {input_ids.shape}, start_ids.shape: {start_ids.shape}, end_ids.shape: {end_ids.shape}"
#          #  )
#          bert_outputs = self.transformer(input_ids=input_ids,
#                                          attention_mask=attention_mask,
#                                          token_type_ids=token_type_ids)
#
#          seq_out = bert_outputs[0]
#
#          seq_out = self.mid_linear(seq_out)
#
#          start_logits = self.start_fc(seq_out)
#          end_logits = self.end_fc(seq_out)
#          #  logger.warning(
#          #      f"start_logits.shape: {start_logits.shape}, end_logits.shape: {end_logits.shape}"
#          #  )
#
#          out = (
#              start_logits,
#              end_logits,
#          )
#
#          if start_ids is not None and end_ids is not None:  #and self.training:
#
#              start_logits = start_logits.view(-1, self.num_labels)
#              end_logits = end_logits.view(-1, self.num_labels)
#
#              # 去掉 padding 部分的标签，计算真实 loss
#              active_loss = attention_mask.view(-1) == 1
#              active_start_logits = start_logits[active_loss]
#              active_end_logits = end_logits[active_loss]
#
#              active_start_labels = start_ids.view(-1)[active_loss]
#              active_end_labels = end_ids.view(-1)[active_loss]
#
#              if pseudo is not None:
#                  seq_len = input_ids.size(1)  # 512
#                  # (batch,)
#                  start_loss = self.criterion(start_logits,
#                                              start_ids.view(-1)).view(
#                                                  -1, seq_len).mean(dim=-1)
#                  end_loss = self.criterion(end_logits, end_ids.view(-1)).view(
#                      -1, seq_len).mean(dim=-1)
#
#                  # nums of pseudo data
#                  pseudo_nums = pseudo.sum().item()
#                  total_nums = input_ids.shape[0]
#
#                  # learning parameter
#                  rate = torch.sigmoid(self.loss_weight)
#                  if pseudo_nums == 0:
#                      start_loss = start_loss.mean()
#                      end_loss = end_loss.mean()
#                  else:
#                      if total_nums == pseudo_nums:
#                          start_loss = (rate * pseudo *
#                                        start_loss).sum() / pseudo_nums
#                          end_loss = (rate * pseudo *
#                                      end_loss).sum() / pseudo_nums
#                      else:
#                          start_loss = (rate*pseudo*start_loss).sum() / pseudo_nums \
#                                       + ((1 - rate) * (1 - pseudo) * start_loss).sum() / (total_nums - pseudo_nums)
#                          end_loss = (rate*pseudo*end_loss).sum() / pseudo_nums \
#                                       + ((1 - rate) * (1 - pseudo) * end_loss).sum() / (total_nums - pseudo_nums)
#              else:
#                  seq_len = input_ids.size(1)  # 512
#                  start_loss = self.criterion(start_logits,
#                                              start_ids.view(-1)).view(
#                                                  -1, seq_len).mean(dim=-1)
#                  end_loss = self.criterion(end_logits, end_ids.view(-1)).view(
#                      -1, seq_len).mean(dim=-1)
#                  start_loss = start_loss.mean()
#                  end_loss = end_loss.mean()
#                  #  start_loss = self.criterion(active_start_logits,
#                  #                              active_start_labels)
#                  #  end_loss = self.criterion(active_end_logits, active_end_labels)
#                  #  logger.info(
#                  #      f"start_loss.shape: {start_loss.shape}, active_start_logits.shape: {active_start_logits.shape}, active_start_labels.shape: {active_start_labels.shape}"
#                  #  )
#                  #  logger.info(
#                  #      f"end_loss.shape: {end_loss.shape}, active_end_logits.shape: {active_end_logits.shape}, active_end_labels.shape: {active_end_labels.shape}"
#                  #  )
#
#              loss = start_loss + end_loss
#
#              out = (loss, ) + out
#
#          return out
#
#      @classmethod
#      def decode_ents(cls, start_logits, end_logits):
#
#          #  logger.info(f"start_logits.shape: {start_logits.shape}")
#          #  logger.info(f"end_logits.shape: {end_logits.shape}")
#          start_logits = start_logits[:, 1:-1]
#          end_logits = end_logits[:, 1:-1]
#          start_preds = np.argmax(start_logits, -1)
#          end_preds = np.argmax(end_logits, -1)
#          #  logger.info(f"start_preds: {start_preds.shape} {start_preds[:3,:]}")
#          #  logger.info(f"end_preds: {end_preds.shape} {end_preds[:3,:]}")
#
#          final_predict_ents = []
#          for start_pred, end_pred in zip(start_preds, end_preds):
#              predict_ents = defaultdict(list)
#              for i, s_type in enumerate(start_pred):
#                  if s_type == 0:
#                      continue
#                  for j, e_type in enumerate(end_pred[i:]):
#                      if s_type == e_type:
#                          s = i
#                          e = j + i
#                          predict_ents[s_type].append((s, e))
#                          break
#              final_predict_ents.append(predict_ents)
#          #  logger.warning(f"final_predict_ents: {final_predict_ents}")
#
#          return final_predict_ents


# ------------------------------ TaskRunner ------------------------------
class NerRunner(TaskRunner):
    """
    任务专属模型定义
    """
    def __init__(self, task_args, ner_labels):
        super(NerRunner, self).__init__(**task_args.to_dict())
        logger.warning(f"ner_labels: {ner_labels}")
        self.ner_labels = ner_labels
        self.num_labels = len(self.ner_labels) + 1
        self.label2id = {x: i + 1 for i, x in enumerate(ner_labels)}
        self.id2label = {i + 1: x for i, x in enumerate(ner_labels)}
        self.type_weights = np.ones(self.num_labels)

        model_args = task_args.model_args
        #  self.model = SpanModel(
        self.model = BertSpanModel(
            model_name_or_path=model_args.model_name_or_path
            if model_args.model_name_or_path else model_args.checkpoint_path,
            num_labels=self.num_labels)

        self.all_ccm_list = []

    def forward(self, *args, **kwargs):
        kwargs = generate_method_kwargs_from_arguments(self.model.__class__,
                                                       "forward",
                                                       dict(**kwargs))
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss, start_logits, end_logits = outputs

        self.log('train_loss', loss, on_step=True)
        #  self.log('lr', self.hparams.lr, on_step=True)

        return OrderedDict({
            "loss": loss,
            'start_logits': start_logits,
            'end_logits': end_logits
        })

    def calculate_tp_fp_fn(self, golden_tags, pred_tags):
        """
        [(start, end), ...]
        """
        tp, fp, fn = 0, 0, 0
        for pred_tag in pred_tags:
            flag = 0
            for entity_gt in golden_tags:
                if pred_tag[0] == entity_gt[0] and pred_tag[1] == entity_gt[1]:
                    flag = 1
                    tp += 1
                    break
            if flag == 0:
                fp += 1

        fn = len(golden_tags) - tp

        return np.array([tp, fp, fn])

    def category_confuse_matrix(self, golden_tags, pred_tags):
        """
        {category_id: [(start, end), ...]}
        """
        ccm = {}
        for c_id, c in self.id2label.items():
            if c_id not in ccm:
                ccm[c_id] = np.array([0, 0, 0])
            ccm[c_id] += self.calculate_tp_fp_fn(golden_tags.get(c_id, []),
                                                 pred_tags.get(c_id, []))

        return ccm

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        val_loss, start_logits, end_logits = outputs
        #  logger.warning(f"{val_loss}")
        self.log('val_loss', val_loss, on_step=True)

        start_probs = F.softmax(start_logits, -1)
        end_probs = F.softmax(end_logits, -1)
        #  logger.info(f"start_probs: {start_probs}")
        start_probs = start_probs.cpu().numpy()
        end_probs = end_probs.cpu().numpy()
        batch_lens = [am.sum() for am in batch['attention_mask']]
        batch_pred_tags = self.model.decode_ents(start_probs,
                                                 end_probs,
                                                 batch_lens=batch_lens)

        batch_tags = batch['tags']
        batch_char2tokens = batch['char2token']

        batch_golden_tags = []
        for tags, char2token in zip(batch_tags, batch_char2tokens):
            golden_tags = defaultdict(list)
            for tag in tags:
                c = tag['category']
                s = tag['start']
                m = tag['mention']
                e = s + len(m) - 1
                s = char2token[s]
                e = char2token[e]
                item = (s, e)
                golden_tags[self.label2id[c]].append(item)
            batch_golden_tags.append(golden_tags)

        all_ccm = []
        for golden_tags, pred_tags in zip(batch_golden_tags, batch_pred_tags):
            ccm = self.category_confuse_matrix(golden_tags, pred_tags)
            all_ccm.append(ccm)

        self.all_ccm_list.append(all_ccm)
        category_confuse_matrix = {
            c_id: np.array([0, 0, 0])
            for c_id in self.id2label.keys()
        }
        for all_ccm in self.all_ccm_list:
            for ccm in all_ccm:
                for c_id, x in ccm.items():
                    category_confuse_matrix[c_id] += x
        micro_metrics = np.zeros(3)
        category_p_r_f1 = {}
        for c_id, (tp, fp, fn) in category_confuse_matrix.items():
            p_r_f1 = get_p_r_f1(tp, fp, fn)
            category_p_r_f1[self.id2label[c_id]] = (p_r_f1, (tp, fp, fn))
            micro_metrics += p_r_f1 * self.type_weights[c_id - 1]
        micro_metrics /= self.num_labels
        val_precision, val_recall, val_f1 = micro_metrics
        self.log("val_precision", val_precision, on_step=True)
        self.log("val_recall", val_recall, on_step=True)
        self.log("val_f1", val_f1, on_step=True)

        return OrderedDict({
            'val_loss': val_loss,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'category_confuse_matrix': all_ccm
        })

    def show_val_results(self, category_p_r_f1):
        num_categories = self.num_labels - 1

        c_p_r_f1 = sorted(category_p_r_f1.items(),
                          key=lambda x: x[1][0][2],
                          reverse=True)
        max_key_len = max(max([len(c.encode('gbk')) + 1 for c, _ in c_p_r_f1]),
                          5)
        #  max_key_len = 30

        title = f"{' '*max_key_len}   P     R     F1   "
        title += " |   tp |   fp |   fn | true "
        title_len = len(title.encode('gbk'))

        #  n_english = len(re.findall('[^\u4e00-\u9fa5]', title))
        #  title_len = (len(title) - n_english) * 2 + n_english

        logger.info(f"{'=' * title_len}")
        logger.info(title)
        logger.info(f"{'-' * title_len}")

        total_tp = 0
        total_fp = 0
        total_fn = 0

        micro_metrics = np.zeros(3)
        for c, ((p, r, f1), (tp, fp, fn)) in c_p_r_f1:
            #  disp_key = c[:16]
            disp_key = c
            disp_key += ' ' * (max_key_len - len(disp_key.encode('gbk')))
            #  n_english = len(re.findall('[^\u4e00-\u9fa5]', disp_key))
            #  disp_key += ' ' * ((max_key_len -
            #                      (len(disp_key) - n_english) * 2 + n_english))
            info = f"{disp_key} | {p:.3f} {r:.3f} {f1:.3f}"
            info += f" | {tp:4d} | {fp:4d} | {fn:4d} | {tp+fn:4d}"
            logger.info(info)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            c_id = self.label2id[c]
            p_r_f1 = np.array([p, r, f1])
            micro_metrics += p_r_f1 * self.type_weights[c_id - 1]

        logger.info(f"{'-' * title_len}")

        total_pred = total_tp + total_fp
        total_true = total_tp + total_fn

        micro_metrics /= num_categories
        micro_acc, micro_recall, micro_f1 = micro_metrics
        info = f"Micro{' '*(max_key_len - 6)} | {micro_acc:.3f} {micro_recall:.3f} {micro_f1:.3f}"  #" - loss: {results['loss']:.4f}"
        info += f" | {total_tp:4d} | {total_fp:4d} | {total_fn} | {total_true:4d}"
        logger.info(info)

        macro_acc = total_tp / total_pred if total_pred > 0.0 else 0.0
        macro_recall = total_tp / total_true if total_true > 0.0 else 0.0
        macro_f1 = 2 * macro_acc * macro_recall / (
            macro_acc + macro_recall) if (macro_acc +
                                          macro_recall) > 0.0 else 0.0
        info = f"Macro{' '*(max_key_len - 6)} | {macro_acc:.3f} {macro_recall:.3f} {macro_f1:.3f}"  #" - loss: {results['loss']:.4f}"
        logger.info(info)

    def validation_epoch_end(self, outputs):
        x = [out['val_loss'].shape for out in outputs]
        val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)
        self.log('val_loss', val_loss, on_epoch=True)

        category_confuse_matrix = {
            c_id: np.array([0, 0, 0])
            for c_id in self.id2label.keys()
        }
        for out in outputs:
            all_ccm = out['category_confuse_matrix']
            for ccm in all_ccm:
                for c_id, x in ccm.items():
                    category_confuse_matrix[c_id] += x

        micro_metrics = np.zeros(3)
        category_p_r_f1 = {}
        for c_id, (tp, fp, fn) in category_confuse_matrix.items():
            p_r_f1 = get_p_r_f1(tp, fp, fn)
            category_p_r_f1[self.id2label[c_id]] = (p_r_f1, (tp, fp, fn))
            micro_metrics += p_r_f1 * self.type_weights[c_id - 1]

        micro_metrics /= self.num_labels - 1

        val_precision, val_recall, val_f1 = micro_metrics
        #  logger.info(f"category_p_r_f1: {category_p_r_f1}")
        #  logger.info(f"micro metrics: {micro_metrics}")
        #  logger.info(
        #      f"val_precision: {val_precision:.4f}, val_recall: {val_recall:.4f}, val_f1: {val_f1:.4f}"
        #  )
        self.log("val_precision", val_precision, on_epoch=True)
        self.log("val_recall", val_recall, on_epoch=True)
        self.log("val_f1", val_f1, on_epoch=True)

        self.show_val_results(category_p_r_f1)

        #  logger.warning(
        #      f"trainer.callback_metrics: {self.trainer.callback_metrics}")
        eval_outputs = {
            'val_loss': val_loss,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        }
        self.save_best_model(eval_outputs)

    def test_step(self, batch, batch_idx):
        test_kwargs = {**batch}
        test_kwargs['start_ids'] = None
        test_kwargs['end_ids'] = None
        test_kwargs['tags'] = None
        #  start_logits, end_logits = self.forward(**batch)
        start_logits, end_logits = self.forward(**test_kwargs)

        start_probs = F.softmax(start_logits, -1)
        end_probs = F.softmax(end_logits, -1)
        start_probs = start_probs.cpu().numpy()
        end_probs = end_probs.cpu().numpy()

        batch_lens = [am.sum() for am in batch['attention_mask']]
        batch_pred_tags = self.model.decode_ents(start_probs,
                                                 end_probs,
                                                 batch_lens=batch_lens)

        batch_token2chars = batch['token2char']
        for pred_tags, token2char in zip(batch_pred_tags, batch_token2chars):
            for c, tags in pred_tags.items():
                pred_tags[c] = [(token2char[s], token2char[e + 1] - 1)
                                for s, e in tags]

        return OrderedDict({
            'preds': batch_pred_tags,
            'start_logits': start_logits,
            'end_logits': end_logits,
        })

    def test_epoch_end(self, outputs):

        test_preds = np.concatenate([out['preds'] for out in outputs])
        logger.info(f"test_preds: {test_preds.shape}")

        test_dataset = self.test_dataloader.dataloader.dataset
        seg_spans = test_dataset.seg_spans

        assert len(seg_spans) == test_preds.shape[
            0], f"len(seg_spans): {len(seg_spans)} == test_preds.shape[0]: {test_preds.shape[0]}"

        test_ents = defaultdict(list)
        for (guid, s_seg, e_seg), dict_ents in zip(seg_spans, test_preds):
            shift_ents = []
            for c, ents in dict_ents.items():
                ents = [(c, s + s_seg, e + s_seg) for s, e in ents
                        if s >= 0 and e >= 0]

                if ents:
                    shift_ents.extend(ents)

            test_ents[guid].append(shift_ents)

        # 合并去重
        for guid, ents in test_ents.items():
            ents = [x for x in ents if x]
            unique_ents = []
            ent_keys = []
            if len(ents) > 0:
                ents = np.concatenate(ents).tolist()
                for ent in ents:
                    key = str(ent)
                    if key not in ent_keys:
                        ent_keys.append(key)
                        unique_ents.append(ent)
            test_ents[guid] = unique_ents
        #  for guid, ents in test_ents.items():
        #      unique_ents = []
        #      ent_keys = []
        #      for ent in ents:
        #          key = str(ent)
        #          logger.info(f"key: {key}")
        #          if key not in ent_keys:
        #              ent_keys.append(key)
        #              unique_ents.append(ent)
        #          else:
        #              logger.warning(f"Duplicate {key}")
        #      test_ents[guid] = unique_ents

        #  assert len(test_ents) == len(
        #      test_dataset
        #  ), f"len(test_ents): {len(test_ents)}, len(test_dataset): {len(test_dataset)}"

        final_preds = test_ents

        #  final_start_logits = np.concatenate(
        #      [out['start_logits'] for out in outputs])
        #  logger.info(f"start_logits: {final_start_logits.shape}")
        #  final_start_logits = final_start_logits.tolist()
        #
        #  final_end_logits = np.concatenate(
        #      [out['end_logits'] for out in outputs])
        #  logger.info(f"end_logits: {final_end_logits.shape}")
        #  final_end_logits = final_end_logits.tolist()

        self.test_results = {
            'preds': final_preds,
            #  'start_logits': final_start_logits,
            #  'end_logits': final_end_logits
        }


# ------------------------------ Task ------------------------------
class NerTask(BaseTask):
    #  def __init__(self, *args, **kwargs):
    #      super(NerTask, self).__init__(*args, **kwargs)

    def __init__(self, args: Type[TaskArguments], data: Type[TaskData],
                 ner_labels: list):
        runner = NerRunner(args, ner_labels)
        super(NerTask, self).__init__(args, data, runner)

    def execute(self, *args, **kwargs):
        model_args = self.model_args
        data_args = self.data_args
        training_args = self.training_args
        remaining_args = self.remaining_args

        super(NerTask, self).execute(*args, **kwargs)

    def generate_submission(self):
        logger.warning(f"NerTask.generate_submission().")
