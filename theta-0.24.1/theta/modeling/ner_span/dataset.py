#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm

from ...utils import seg_generator
from ...utils.multiprocesses import (barrier_leader_process,
                                     barrier_member_processes,
                                     is_multi_processes)
from ..ner_utils import InputExample
from ..trainer import common_batch_encode, common_to_tensors


class InputFeature(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask, token_type_ids, input_len,
                 token_offsets, start_ids, end_ids, subjects, text):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.token_offsets = token_offsets

        self.start_ids = start_ids
        self.end_ids = end_ids
        self.subjects = subjects
        self.text = text

    def __repr__(self):
        return str(self.to_json_string())

    def __eq__(self, other):
        return self.input_ids == other.input_ids and \
            self.attention_mask == other.attention_mask and \
            self.token_type_ids == other.token_type_ids and \
            self.start_ids == other.start_ids and \
            self.end_ids == other.end_ids and \
            self.subjects == other.subjects and \
            self.input_len == other.input_len

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


#  def encode_examples_undone(examples,
#                      label2id,
#                      tokenizer,
#                      max_seq_length,
#                      seg_len=0,
#                      seg_backoff=0):
#      all_features = []
#      texts = [e.text_a for e in examples]
#      all_subjects = [e.labels for e in examples]
#
#      all_seg_tokens = []
#      all_seg_subjects = []
#      #  all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets
#      for e in tqdm(examples, desc="Text to tokens"):
#          text = e.text_a
#          subjects = e.labels
#          encodes = tokenizer.encode(text, add_special_tokens=False)
#          #  return {
#          #      'tokens': text_tokens.tokens,
#          #      'offsets': text_tokens.offsets,
#          #      'token2char': token2char,
#          #      'char2token': char2token,
#          #      'ids': text_tokens.ids,
#          #      'attention_mask': text_tokens.attention_mask,
#          #      'type_ids': text_tokens.type_ids,
#          #  }
#          tokens = encodes['tokens']
#          char2token = encodes['char2token']
#          num_tokens = len(tokens)
#          seg_size = seg_len - seg_backoff
#
#          for i in range(0, num_tokens, seg_size):
#              seg_start = seg_size * i
#              seg_end = min(seg_size * i + seg_len, num_tokens)
#              seg_tokens = tokens[seg_start:seg_end]
#              seg_subjects = [
#                  (c, char2token[s] - seg_start, char2token[e] - seg_start)
#                  for c, s, e in subjects
#                  if char2token[s] >= seg_start and char2token[e] < seg_end
#              ]
#              all_seg_tokens.append([seg_tokens])
#              all_seg_subjects.append(seg_subjects)
#
#      def encode_subjects(subjects, num_tokens, char2token):
#          #  logger.warning(f"num_tokens: {num_tokens}")
#          start_ids = [label2id['[unused1]']] * num_tokens
#          end_ids = [label2id['[unused1]']] * num_tokens
#          subjects_id = []
#          if subjects:
#              for subject in subjects:
#                  #  logger.debug(f"subject: {subject}")
#                  label = subject[0]
#                  start = subject[1]
#                  end = subject[2]
#
#                  if start < 0 or start >= len(char2token):
#                      continue
#                  if end < 0 or end >= len(char2token):
#                      continue
#                  start = char2token[start]
#                  end = char2token[end]
#
#                  start_ids[start] = label2id[label]
#                  end_ids[end] = label2id[label]
#                  subjects_id.append((label2id[label], start, end))
#          return subjects_id, start_ids, end_ids
#
#      all_subjects_ids = []
#      all_start_ids = []
#      all_end_ids = []
#      for input_ids, attention_mask, token_type_ids, subjects, char2token in zip(
#              all_input_ids, all_attention_mask, all_token_type_ids,
#              all_subjects, all_char2token):
#          num_tokens = len(input_ids)
#          subjects_id, start_ids, end_ids = encode_subjects(
#              subjects, num_tokens, char2token)
#
#          assert len(input_ids) == max_seq_length
#          assert len(attention_mask) == max_seq_length
#          assert len(token_type_ids) == max_seq_length
#          assert len(start_ids) == max_seq_length
#          assert len(end_ids) == max_seq_length
#
#          all_subjects_ids.append(subjects_id)
#          all_start_ids.append(start_ids)
#          all_end_ids.append(end_ids)
#
#      logger.debug(f"all_input_ids.shape: {np.array(all_input_ids).shape}")
#      logger.debug(
#          f"all_attention_mask.shape: {np.array(all_attention_mask).shape}")
#      logger.debug(
#          f"all_token_type_ids.shape: {np.array(all_token_type_ids).shape}")
#      logger.debug(f"all_start_ids.shape: {np.array(all_start_ids).shape}")
#      logger.debug(f"all_end_ids.shape: {np.array(all_end_ids).shape}")
#      logger.debug(f"all_subjects_ids.shape: {np.array(all_subjects_ids).shape}")
#      logger.debug(f"all_input_lens.shape: {np.array(all_input_lens).shape}")
#      assert np.array(all_input_ids).shape[1] == max_seq_length
#      assert np.array(all_attention_mask).shape[1] == max_seq_length
#      assert np.array(all_token_type_ids).shape[1] == max_seq_length
#
#      #  all_input_ids = torch.from_numpy(np.array(all_input_ids, dtype=np.int64))
#      #  all_attention_mask = torch.from_numpy(
#      #      np.array(all_attention_mask, dtype=np.int64))
#      #  all_token_type_ids = torch.from_numpy(
#      #      np.array(all_token_type_ids, dtype=np.int64))
#      #  all_input_lens = torch.from_numpy(np.array(all_input_lens, dtype=np.int64))
#      #  all_token_offsets = torch.from_numpy(
#      #      np.array(all_token_offsets, dtype=np.int64))
#      all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets = common_to_tensors(
#          all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens,
#          all_token_offsets)
#
#      all_start_ids = torch.from_numpy(np.array(all_start_ids, dtype=np.int64))
#      all_end_ids = torch.from_numpy(np.array(all_end_ids, dtype=np.int64))
#
#      all_features = [
#          InputFeature(input_ids=input_ids,
#                       attention_mask=attention_mask,
#                       token_type_ids=token_type_ids,
#                       input_len=input_len,
#                       token_offsets=token_offsets,
#                       start_ids=start_ids,
#                       end_ids=end_ids,
#                       subjects=subjects_ids,
#                       text=text)
#          for input_ids, attention_mask, token_type_ids, start_ids, end_ids,
#          subjects_ids, input_len, token_offsets, text in zip(
#              all_input_ids, all_attention_mask, all_token_type_ids,
#              all_start_ids, all_end_ids, all_subjects_ids, all_input_lens,
#              all_token_offsets, texts)
#      ]
#
#      return all_features
#
#      return all_features

#  def __encode_examples(texts, all_subjects, label2id, tokenizer,
#                        max_seq_length):
#
#      #  num_labels = len(label2id)
#      #  texts = [e.text_a[:max_seq_length - 2] for e in examples]
#      #
#      #  all_encodes = tokenizer.batch_encode(texts, add_special_tokens=True)
#      #  all_tokens = all_encodes['tokens']
#      #  all_token_offsets = all_encodes['offsets']
#      #  all_token2char = all_encodes['token2char']
#      #  all_char2token = all_encodes['char2token']
#      #
#      #  all_input_ids = all_encodes['ids']
#      #  all_attention_mask = all_encodes['attention_mask']
#      #  all_token_type_ids = all_encodes['type_ids']
#      #
#      #  all_input_lens = [len(tokens) for tokens in all_tokens]
#      #
#      #  all_padding_lens = [max_seq_length - n for n in all_input_lens]
#      #  for i, (input_ids, attention_mask, token_type_ids, token2char,
#      #          token_offsets, padding_length) in enumerate(
#      #              zip(all_input_ids, all_attention_mask, all_token_type_ids,
#      #                  all_token2char, all_token_offsets, all_padding_lens)):
#      #      all_input_ids[i] = input_ids + [0] * padding_length
#      #      all_attention_mask[i] = attention_mask + [0] * padding_length
#      #      all_token_type_ids[i] = token_type_ids + [0] * padding_length
#      #      all_token2char[i] = token2char + [0] * padding_length
#      #      all_token_offsets[i] = token_offsets + [(0, 0)] * padding_length
#
#      num_labels = len(label2id)
#
#      all_tokens, all_token2char, all_char2token, all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets = common_batch_encode(
#          texts, label2id, tokenizer, max_seq_length)
#
#      def encode_subjects(subjects, num_tokens, char2token):
#          #  logger.warning(f"num_tokens: {num_tokens}")
#          start_ids = [label2id['[unused1]']] * num_tokens
#          end_ids = [label2id['[unused1]']] * num_tokens
#          subjects_id = []
#          if subjects:
#              for subject in subjects:
#                  #  logger.debug(f"subject: {subject}")
#                  label = subject[0]
#                  start = subject[1]
#                  end = subject[2]
#
#                  if start < 0 or start >= len(char2token):
#                      continue
#                  if end < 0 or end >= len(char2token):
#                      continue
#                  start = char2token[start]
#                  end = char2token[end]
#
#                  start_ids[start] = label2id[label]
#                  end_ids[end] = label2id[label]
#                  subjects_id.append((label2id[label], start, end))
#          return subjects_id, start_ids, end_ids
#
#      all_subjects_ids = []
#      all_start_ids = []
#      all_end_ids = []
#      for input_ids, attention_mask, token_type_ids, subjects, char2token in zip(
#              all_input_ids, all_attention_mask, all_token_type_ids,
#              all_subjects, all_char2token):
#          num_tokens = len(input_ids)
#          subjects_id, start_ids, end_ids = encode_subjects(
#              subjects, num_tokens, char2token)
#
#          assert len(input_ids) == max_seq_length
#          assert len(attention_mask) == max_seq_length
#          assert len(token_type_ids) == max_seq_length
#          assert len(start_ids) == max_seq_length
#          assert len(end_ids) == max_seq_length
#
#          all_subjects_ids.append(subjects_id)
#          all_start_ids.append(start_ids)
#          all_end_ids.append(end_ids)
#
#      logger.debug(f"all_input_ids.shape: {np.array(all_input_ids).shape}")
#      logger.debug(
#          f"all_attention_mask.shape: {np.array(all_attention_mask).shape}")
#      logger.debug(
#          f"all_token_type_ids.shape: {np.array(all_token_type_ids).shape}")
#      logger.debug(f"all_start_ids.shape: {np.array(all_start_ids).shape}")
#      logger.debug(f"all_end_ids.shape: {np.array(all_end_ids).shape}")
#      logger.debug(f"all_subjects_ids.shape: {np.array(all_subjects_ids).shape}")
#      logger.debug(f"all_input_lens.shape: {np.array(all_input_lens).shape}")
#      assert np.array(all_input_ids).shape[1] == max_seq_length
#      assert np.array(all_attention_mask).shape[1] == max_seq_length
#      assert np.array(all_token_type_ids).shape[1] == max_seq_length
#
#      #  all_input_ids = torch.from_numpy(np.array(all_input_ids, dtype=np.int64))
#      #  all_attention_mask = torch.from_numpy(
#      #      np.array(all_attention_mask, dtype=np.int64))
#      #  all_token_type_ids = torch.from_numpy(
#      #      np.array(all_token_type_ids, dtype=np.int64))
#      #  all_input_lens = torch.from_numpy(np.array(all_input_lens, dtype=np.int64))
#      #  all_token_offsets = torch.from_numpy(
#      #      np.array(all_token_offsets, dtype=np.int64))
#      all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets = common_to_tensors(
#          all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens,
#          all_token_offsets)
#
#      all_start_ids = torch.from_numpy(np.array(all_start_ids, dtype=np.int64))
#      all_end_ids = torch.from_numpy(np.array(all_end_ids, dtype=np.int64))
#
#      all_features = [
#          InputFeature(input_ids=input_ids,
#                       attention_mask=attention_mask,
#                       token_type_ids=token_type_ids,
#                       input_len=input_len,
#                       token_offsets=token_offsets,
#                       start_ids=start_ids,
#                       end_ids=end_ids,
#                       subjects=subjects_ids,
#                       text=text)
#          for input_ids, attention_mask, token_type_ids, start_ids, end_ids,
#          subjects_ids, input_len, token_offsets, text in zip(
#              all_input_ids, all_attention_mask, all_token_type_ids,
#              all_start_ids, all_end_ids, all_subjects_ids, all_input_lens,
#              all_token_offsets, texts)
#      ]
#
#      return all_features

#  enttype_labels = defaultdict(list)


def encode_examples(examples,
                    label2id,
                    tokenizer,
                    max_seq_length,
                    enttype_labels=None,
                    epoch=-1):

    #  num_labels = len(label2id)
    #  texts = [e.text_a[:max_seq_length - 2] for e in examples]
    #
    #  all_encodes = tokenizer.batch_encode(texts, add_special_tokens=True)
    #  all_tokens = all_encodes['tokens']
    #  all_token_offsets = all_encodes['offsets']
    #  all_token2char = all_encodes['token2char']
    #  all_char2token = all_encodes['char2token']
    #
    #  all_input_ids = all_encodes['ids']
    #  all_attention_mask = all_encodes['attention_mask']
    #  all_token_type_ids = all_encodes['type_ids']
    #
    #  all_input_lens = [len(tokens) for tokens in all_tokens]
    #
    #  all_padding_lens = [max_seq_length - n for n in all_input_lens]
    #  for i, (input_ids, attention_mask, token_type_ids, token2char,
    #          token_offsets, padding_length) in enumerate(
    #              zip(all_input_ids, all_attention_mask, all_token_type_ids,
    #                  all_token2char, all_token_offsets, all_padding_lens)):
    #      all_input_ids[i] = input_ids + [0] * padding_length
    #      all_attention_mask[i] = attention_mask + [0] * padding_length
    #      all_token_type_ids[i] = token_type_ids + [0] * padding_length
    #      all_token2char[i] = token2char + [0] * padding_length
    #      all_token_offsets[i] = token_offsets + [(0, 0)] * padding_length

    #  if epoch >= 0 and enttype_labels:
    #      old_examples = copy.deepcopy(examples)
    #      random.seed(8864)
    #      for e in tqdm(examples, desc="Aug entitiesi in epoch {epoch}"):
    #          p0 = 0
    #          text = copy.deepcopy(e.text_a)
    #          new_text = ""
    #          new_labels = []
    #          for i, (label, start, end) in enumerate(e.labels):
    #              new_text += text[p0:start]
    #              if random.randint(0, 1000 - 1) < 666:
    #                  new_start = len(new_text)
    #                  new_mention = text[start:end + 1]
    #                  new_text += new_mention
    #                  new_labels.append(
    #                      (label, new_start, new_start + len(new_mention) - 1))
    #                  p0 = end + 1
    #              else:
    #                  new_start = len(new_text)
    #                  new_idx = random.randint(0, len(enttype_labels[label]) - 1)
    #                  new_mention = enttype_labels[label][new_idx]
    #                  new_text += new_mention
    #                  new_end = new_start + len(new_mention) - 1
    #                  new_labels.append((label, new_start, new_end))
    #                  p0 = end + 1
    #          new_text += text[p0:]
    #          #  logger.debug(
    #          #      f"text: {text} | {[ text[start:end+1] for _, start, end in e.labels]}"
    #          #  )
    #          #  logger.warning(
    #          #      f"new_text: {new_text} | {[ new_text[start:end+1] for _, start, end in new_labels]}"
    #          #  )
    #          e.text_a = new_text
    #          e.labels = new_labels
    #
    #      examples = old_examples + examples
    #      random.sample(examples, len(examples))
    #
    num_labels = len(label2id)
    #  texts = [e.text_a[:max_seq_length - 2] for e in examples]
    texts = [e.text_a for e in examples]

    all_tokens, all_token2char, all_char2token, all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets = common_batch_encode(
        texts, label2id, tokenizer, max_seq_length)

    def encode_subjects(subjects, num_tokens, char2token):
        #  logger.warning(f"num_tokens: {num_tokens}")
        start_ids = [label2id['[unused1]']] * num_tokens
        end_ids = [label2id['[unused1]']] * num_tokens
        subjects_id = []
        if subjects:
            for subject in subjects:
                #  logger.debug(f"subject: {subject}")
                label = subject[0]
                start = subject[1]
                end = subject[2]

                if start < 0 or start >= len(char2token):
                    continue
                if end < 0 or end >= len(char2token):
                    continue
                start = char2token[start]
                end = char2token[end]

                if start >= num_tokens:
                    continue
                if end >= num_tokens:
                    continue

                start_ids[start] = label2id[label]
                end_ids[end] = label2id[label]
                subjects_id.append((label2id[label], start, end))
        return subjects_id, start_ids, end_ids

    all_subjects_ids = []
    all_start_ids = []
    all_end_ids = []
    all_subjects = [e.labels for e in examples]
    for input_ids, attention_mask, token_type_ids, subjects, char2token in zip(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_subjects, all_char2token):
        num_tokens = len(input_ids)
        subjects_id, start_ids, end_ids = encode_subjects(
            subjects, num_tokens, char2token)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length

        all_subjects_ids.append(subjects_id)
        all_start_ids.append(start_ids)
        all_end_ids.append(end_ids)

    logger.debug(f"all_input_ids.shape: {np.array(all_input_ids).shape}")
    logger.debug(
        f"all_attention_mask.shape: {np.array(all_attention_mask).shape}")
    logger.debug(
        f"all_token_type_ids.shape: {np.array(all_token_type_ids).shape}")
    logger.debug(f"all_start_ids.shape: {np.array(all_start_ids).shape}")
    logger.debug(f"all_end_ids.shape: {np.array(all_end_ids).shape}")
    logger.debug(f"all_subjects_ids.shape: {np.array(all_subjects_ids).shape}")
    logger.debug(f"all_input_lens.shape: {np.array(all_input_lens).shape}")
    assert np.array(all_input_ids).shape[1] == max_seq_length
    assert np.array(all_attention_mask).shape[1] == max_seq_length
    assert np.array(all_token_type_ids).shape[1] == max_seq_length

    #  all_input_ids = torch.from_numpy(np.array(all_input_ids, dtype=np.int64))
    #  all_attention_mask = torch.from_numpy(
    #      np.array(all_attention_mask, dtype=np.int64))
    #  all_token_type_ids = torch.from_numpy(
    #      np.array(all_token_type_ids, dtype=np.int64))
    #  all_input_lens = torch.from_numpy(np.array(all_input_lens, dtype=np.int64))
    #  all_token_offsets = torch.from_numpy(
    #      np.array(all_token_offsets, dtype=np.int64))
    all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets = common_to_tensors(
        all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens,
        all_token_offsets)

    all_start_ids = torch.from_numpy(np.array(all_start_ids, dtype=np.int64))
    all_end_ids = torch.from_numpy(np.array(all_end_ids, dtype=np.int64))

    all_features = [
        InputFeature(input_ids=input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids,
                     input_len=input_len,
                     token_offsets=token_offsets,
                     start_ids=start_ids,
                     end_ids=end_ids,
                     subjects=subjects_ids,
                     text=text)
        for input_ids, attention_mask, token_type_ids, start_ids, end_ids,
        subjects_ids, input_len, token_offsets, text in zip(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_start_ids, all_end_ids, all_subjects_ids, all_input_lens,
            all_token_offsets, texts)
    ]

    return all_features


def encode_examples_old(examples, label2id, tokenizer, max_seq_length):

    #  texts = [''.join(e.text_a) for e in examples]
    texts = [e.text_a[:max_seq_length - 2] for e in examples]

    #  outputs = tokenizer.batch_encode_plus(texts,
    #                                        max_length=max_seq_length,
    #                                        add_special_tokens=True,
    #                                        return_tensors='pt',
    #                                        pad_to_max_length=True,
    #                                        return_attention_mask=True,
    #                                        return_token_type_ids=True)
    #  all_input_ids, all_attention_mask, all_token_type_ids = outputs[
    #      'input_ids'], outputs['attention_mask'], outputs['token_type_ids']

    all_tokens = [[tokenizer.cls_token] + tokenizer.tokenize(text) +
                  [tokenizer.sep_token]
                  for text in tqdm(texts, desc="Tokenize")]
    all_input_lens = [len(tokens) for tokens in all_tokens]
    all_input_ids = [
        tokenizer.convert_tokens_to_ids(tokens) for tokens in all_tokens
    ]
    all_attention_mask = [[1] * len(input_ids) for input_ids in all_input_ids]
    all_token_type_ids = [[0] * len(input_ids) for input_ids in all_input_ids]

    all_padding_lens = [max_seq_length - n for n in all_input_lens]
    for i, (input_ids, attention_mask, token_type_ids,
            padding_length) in enumerate(
                zip(all_input_ids, all_attention_mask, all_token_type_ids,
                    all_padding_lens)):
        all_input_ids[i] = input_ids + [0] * padding_length
        all_attention_mask[i] = attention_mask + [0] * padding_length
        all_token_type_ids[i] = token_type_ids + [0] * padding_length

    def encode_subjects(subjects, num_tokens):
        #  logger.warning(f"num_tokens: {num_tokens}")
        start_ids = [label2id['[unused1]']] * num_tokens
        end_ids = [label2id['[unused1]']] * num_tokens
        subjects_id = []
        if subjects:
            for subject in subjects:
                #  logger.debug(f"subject: {subject}")
                label = subject[0]
                start = subject[1]
                end = subject[2]
                start_ids[start + 1] = label2id[label]
                end_ids[end + 1] = label2id[label]
                subjects_id.append((label2id[label], start, end))
        return subjects_id, start_ids, end_ids

    all_subjects_ids = []
    all_start_ids = []
    all_end_ids = []
    all_subjects = [e.labels for e in examples]
    for input_ids, attention_mask, token_type_ids, subjects in zip(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_subjects):
        num_tokens = len(input_ids)
        subjects_id, start_ids, end_ids = encode_subjects(subjects, num_tokens)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length

        all_subjects_ids.append(subjects_id)
        all_start_ids.append(start_ids)
        all_end_ids.append(end_ids)

    #  all_start_ids = torch.tensor(all_start_ids, dtype=torch.long)
    #  all_end_ids = torch.tensor(all_end_ids, dtype=torch.long)
    #  all_input_lens = torch.tensor(all_input_lens, dtype=torch.long)

    #  all_input_lens = torch.tensor(
    #      [len(input_ids) for input_ids in all_input_ids], dtype=torch.long)

    logger.debug(f"all_input_ids.shape: {np.array(all_input_ids).shape}")
    logger.debug(
        f"all_attention_mask.shape: {np.array(all_attention_mask).shape}")
    logger.debug(
        f"all_token_type_ids.shape: {np.array(all_token_type_ids).shape}")
    logger.debug(f"all_start_ids.shape: {np.array(all_start_ids).shape}")
    logger.debug(f"all_end_ids.shape: {np.array(all_end_ids).shape}")
    logger.debug(f"all_subjects_ids.shape: {np.array(all_subjects_ids).shape}")
    logger.debug(f"all_input_lens.shape: {np.array(all_input_lens).shape}")
    assert np.array(all_input_ids).shape[1] == max_seq_length
    assert np.array(all_attention_mask).shape[1] == max_seq_length
    assert np.array(all_token_type_ids).shape[1] == max_seq_length

    all_features = [
        InputFeature(input_ids=input_ids,
                     input_mask=attention_mask,
                     segment_ids=token_type_ids,
                     start_ids=start_ids,
                     end_ids=end_ids,
                     subjects=subjects_ids,
                     input_len=input_len,
                     token_offsets=token_offsets)
        for input_ids, attention_mask, token_type_ids, start_ids,
        end_ids, subjects_ids, input_len, token_offsets in zip(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_start_ids, all_end_ids, all_subjects_ids, all_input_lens,
            all_token_offsets)
    ]
    return (all_features, {
        'input_ids':
        torch.tensor(all_input_ids, dtype=torch.long),
        'attention_mask':
        torch.tensor(all_attention_mask, dtype=torch.long),
        'token_type_ids':
        torch.tensor(all_token_type_ids, dtype=torch.long),
        'start_ids':
        torch.tensor(all_start_ids, dtype=torch.long),
        'end_ids':
        torch.tensor(all_end_ids, dtype=torch.long),
        "subjects_ids":
        all_subjects_ids,
        "input_lens":
        torch.tensor(all_input_lens, dtype=torch.long),
    })


def examples_to_dataset(examples, label2id, tokenizer, max_seq_length):
    all_features, outputs = encode_examples(examples, label2id, tokenizer,
                                            max_seq_length)

    #  dataset_old, features_old = examples_to_dataset_old(
    #      examples, label2id, tokenizer, max_seq_length)
    #
    #  for x, x1 in zip(all_features, features_old):
    #      if x != x1:
    #          logger.info(f"input_ids: {len(x.input_ids)}, {x.input_ids}")
    #          logger.info(f"input_mask: {len(x.input_mask)}, {x.input_mask}")
    #          logger.info(f"segment_ids: {len(x.segment_ids)}, {x.segment_ids}")
    #          logger.info(f"start_ids: {len(x.start_ids)}, {x.start_ids}")
    #          logger.info(f"end_ids: {len(x.end_ids)}, {x.end_ids}")
    #          logger.info(f"subjects: {x.subjects}")
    #          logger.info(f"input_len: {x.input_len}")
    #
    #          if x.input_ids != x1.input_ids:
    #              logger.warning(
    #                  f"input_ids: {len(x1.input_ids)}, {x1.input_ids}")
    #          if x.input_mask != x1.input_mask:
    #              logger.warning(
    #                  f"input_mask: {len(x1.input_mask)}, {x1.input_mask}")
    #          if x.segment_ids != x1.segment_ids:
    #              logger.warning(
    #                  f"segment_ids: {len(x1.segment_ids)}, {x1.segment_ids}")
    #          if x.start_ids != x1.start_ids:
    #              logger.warning(
    #                  f"start_ids: {len(x1.start_ids)}, {x1.start_ids}")
    #          if x.end_ids != x1.end_ids:
    #              logger.warning(f"end_ids: {len(x1.end_ids)}, {x1.end_ids}")
    #          if x.subjects != x1.subjects:
    #              logger.warning(f"subjects: {x1.subjects}")
    #          if x.input_len != x1.input_len:
    #              logger.warning(f"input_len: {x1.input_len}")

    dataset = TensorDataset(
        outputs['input_ids'],
        outputs['attention_mask'],
        outputs['token_type_ids'],
        outputs['start_ids'],
        outputs['end_ids'],
        outputs['input_lens'],
    )

    assert len(dataset) == len(all_features)

    return dataset, all_features


def convert_examples_to_features(
    examples,
    label2id,
    tokenizer,
    max_seq_length,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for ex_index, e in enumerate(tqdm(examples, desc="Convert examples")):
        textlist = e.text_a
        subjects = e.labels

        #  logger.warning(f"textlist: {textlist}")
        tokens = tokenizer.tokenize(textlist)
        #  tokens = tokenizer.tokenize(''.join(textlist))[1:-1]

        #  logger.debug(f"tokens: {tokens}")
        #  logger.debug(f"textlist: {textlist}")
        #  logger.warning(
        #      f"tokens: {len(tokens)} - {tokens[:10]} - {tokens[-10:]}")
        #  logger.warning(
        #      f"textlist: {len(textlist)} - {textlist[:10]} - {textlist[-10:]}")
        #  assert len(tokens) == len(textlist)

        #  logger.debug(f"len(tokens): {len(tokens)}")
        start_ids = [0] * len(tokens)
        end_ids = [0] * len(tokens)
        subjects_id = []
        if subjects:
            for subject in subjects:
                #  logger.debug(f"subject: {subject}")
                label = subject[0]
                start = subject[1]
                end = subject[2]
                #  logger.warning(
                #      f"len(tokens): {len(tokens)}, subject: {subject}")
                start_ids[start] = label2id[label]
                end_ids[end] = label2id[label]
                subjects_id.append((label2id[label], start, end))
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            start_ids = start_ids[:(max_seq_length - special_tokens_count)]
            end_ids = end_ids[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        start_ids += [0]
        end_ids += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            tokens += [cls_token]
            start_ids += [0]
            end_ids += [0]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            start_ids = [0] + start_ids
            end_ids = [0] + end_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] *
                          padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] *
                           padding_length) + segment_ids
            start_ids = ([0] * padding_length) + start_ids
            end_ids = ([0] * padding_length) + end_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            start_ids += ([0] * padding_length)
            end_ids += ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length

        if ex_index < 3:
            logger.debug("*** Example ***")
            logger.debug(f"guid: {e.guid}")
            logger.debug(f"tokens: {' '.join([str(x) for x in tokens])}")
            logger.debug(f"input_ids: {' '.join([str(x) for x in input_ids])}")
            logger.debug(
                f"input_mask: {' '.join([str(x) for x in input_mask])}")
            logger.debug(
                f"segment_ids: {' '.join([str(x) for x in segment_ids])}")
            logger.info(f"start_ids: {' '.join([str(x) for x in start_ids])}")
            logger.info(f"end_ids: {' '.join([str(x) for x in end_ids])}")

            #  logger.info("*** Example ***")
            #  logger.info(f"guid: %s", e.guid)
            #  logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            #  logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            #  logger.info("input_mask: %s",
            #              " ".join([str(x) for x in input_mask]))
            #  logger.info("segment_ids: %s",
            #              " ".join([str(x) for x in segment_ids]))
            #  logger.info("start_ids: %s" % " ".join([str(x)
            #                                          for x in start_ids]))
            #  logger.info("end_ids: %s" % " ".join([str(x) for x in end_ids]))

        features.append(
            InputFeature(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         start_ids=start_ids,
                         end_ids=end_ids,
                         subjects=subjects_id,
                         input_len=input_len))
    return features


def features_to_dataset(features):
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features],
                                   dtype=torch.long)
    all_start_ids = torch.tensor([f.start_ids for f in features],
                                 dtype=torch.long)
    all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
    all_input_lens = torch.tensor([f.input_len for f in features],
                                  dtype=torch.long)
    #  all_subjects_ids = torch.tensor([f.subjects for f in features],
    #                                  dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_start_ids, all_end_ids, all_input_lens)
    return dataset


def examples_to_dataset_with_cache(args,
                                   examples,
                                   tokenizer,
                                   max_seq_length,
                                   evaluate=False,
                                   alias=""):

    # Make sure only the first process in distributed training process the dataset,
    # and the others will use the cache
    if not evaluate:
        barrier_member_processes(args)

    # Load data features from cache or dataset file
    cached_features_file = Path(
        args.data_dir
    ) / f"cached_ner_{args.dataset_name}{'_' + alias if alias else ''}"
    if args.cache_features and cached_features_file.exists(
    ) and not args.overwrite_cache:
        logger.info(
            f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        logger.info(f"Creating features from dataset file at {args.data_dir}")

        features = convert_examples_to_features(
            examples,
            args.label2id,
            tokenizer,
            max_seq_length=max_seq_length,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            pad_on_left=bool(args.model_type in ['xlnet']),
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token
                                                       ])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.cache_features and is_master_process(args):
            logger.info(
                f"Saving features into cached file {cached_features_file}")
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset,
    # and the others will use the cache
    if not evaluate:
        barrier_leader_process(args)

    dataset = features_to_dataset(features)

    return dataset, features


def examples_to_dataset_old(examples, label2id, tokenizer, max_seq_length):

    features = convert_examples_to_features(
        examples,
        label2id,
        tokenizer,
        max_seq_length=max_seq_length,
        cls_token_at_end=False,
        pad_on_left=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)

    dataset = features_to_dataset(features)

    return dataset, features
