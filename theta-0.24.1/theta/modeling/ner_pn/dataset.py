#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import json
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
                 token_offsets, text, labels, subjects):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.token_offsets = token_offsets
        self.text = text
        self.labels = labels
        self.subjects = subjects

    #  def __repr__(self):
    #      return str(self.to_json_string())
    #
    #  def __eq__(self, other):
    #      return self.input_ids == other.input_ids and \
    #          self.attention_mask == other.attention_mask and \
    #          self.token_type_ids == other.token_type_ids and \
    #          self.labels == other.labels and \
    #          self.subjects == other.subjects and \
    #          self.input_len == other.input_len
    #
    #  def to_dict(self):
    #      """Serializes this instance to a Python dictionary."""
    #      output = copy.deepcopy(self.__dict__)
    #      return output
    #
    #  def to_json_string(self):
    #      """Serializes this instance to a JSON string."""
    #      return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


#  def batch_to_input_data(batch):
#      all_input_ids = torch.stack([e.input_ids for e in batch])
#      all_attention_mask = torch.stack([e.attention_mask for e in batch])
#      all_token_type_ids = torch.stack([e.token_type_ids for e in batch])
#      all_labels = torch.stack([e.labels for e in batch])
#      all_input_lens = torch.stack([e.input_len for e in batch])
#      all_token_offsets = torch.stack([e.token_offsets for e in batch])
#      all_subjects = [e.subjects for e in batch]
#
#      return (all_input_ids, all_attention_mask, all_token_type_ids, all_labels,
#              all_input_lens, all_token_offsets, all_subjects)


def encode_examples(examples, label2id, tokenizer, max_seq_length):
    logger.info(f"Encoding examples...")

    num_labels = len(label2id)
    texts = [e.text_a[:max_seq_length - 2] for e in examples]

    all_tokens, all_token2char, all_char2token, all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets = common_batch_encode(
        texts, label2id, tokenizer, max_seq_length)

    def encode_subjects(subjects, num_tokens, char2token, input_len):

        subjects_id = []
        labels = np.zeros((num_tokens, num_labels * 2), dtype=int).tolist()

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

                if start < input_len - 1 and end < input_len:
                    labels[start][(label2id[label] - 1) * 2] = 1
                    labels[end][(label2id[label] - 1) * 2 + 1] = 1

                    subjects_id.append((label2id[label], start, end))

        return subjects_id, labels

    all_subjects_ids = []
    all_labels = []
    all_subjects = [e.labels for e in examples]
    for input_ids, attention_mask, token_type_ids, subjects, char2token, input_len in tqdm(
            zip(all_input_ids, all_attention_mask, all_token_type_ids,
                all_subjects, all_char2token, all_input_lens),
            desc="encode_subjects"):
        num_tokens = len(input_ids)
        subjects_id, labels = encode_subjects(subjects, num_tokens, char2token,
                                              input_len)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(labels) == max_seq_length

        all_subjects_ids.append(subjects_id)
        all_labels.append(labels)

    logger.debug(f"all_input_ids.shape: {all_input_ids.shape}")
    logger.debug(f"all_attention_mask.shape: {all_attention_mask.shape}")
    logger.debug(f"all_token_type_ids.shape: {all_token_type_ids.shape}")
    logger.debug(f"all_input_lens.shape: {all_input_lens.shape}")
    logger.debug(f"all_labels.shape: {len(all_labels)}")
    logger.debug(f"all_subjects_ids.shape: {len(all_subjects_ids)}")
    assert all_input_ids.shape[1] == max_seq_length
    assert all_attention_mask.shape[1] == max_seq_length
    assert all_token_type_ids.shape[1] == max_seq_length
    #  logger.debug(f"all_input_ids.shape: {np.array(all_input_ids).shape}")
    #  logger.debug(
    #      f"all_attention_mask.shape: {np.array(all_attention_mask).shape}")
    #  logger.debug(
    #      f"all_token_type_ids.shape: {np.array(all_token_type_ids).shape}")
    #  logger.debug(f"all_labels.shape: {np.array(all_labels).shape}")
    #  logger.debug(f"all_subjects_ids.shape: {np.array(all_subjects_ids).shape}")
    #  logger.debug(f"all_input_lens.shape: {np.array(all_input_lens).shape}")
    #  assert np.array(all_input_ids).shape[1] == max_seq_length
    #  assert np.array(all_attention_mask).shape[1] == max_seq_length
    #  assert np.array(all_token_type_ids).shape[1] == max_seq_length

    all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets = common_to_tensors(
        all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens,
        all_token_offsets)
    #  all_input_ids = torch.from_numpy(np.array(all_input_ids, dtype=np.int64))
    #  all_attention_mask = torch.from_numpy(
    #      np.array(all_attention_mask, dtype=np.int64))
    #  all_token_type_ids = torch.from_numpy(
    #      np.array(all_token_type_ids, dtype=np.int64))
    #  all_input_lens = torch.from_numpy(np.array(all_input_lens, dtype=np.int64))
    #  all_token_offsets = torch.from_numpy(
    #      np.array(all_token_offsets, dtype=np.int64))

    all_labels = torch.from_numpy(np.array(all_labels, dtype=np.int64))
    all_subjects_ids = all_subjects_ids

    all_features = [
        InputFeature(input_ids=input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids,
                     input_len=input_len,
                     token_offsets=token_offsets,
                     text=text,
                     labels=labels,
                     subjects=subjects_ids)
        for input_ids, attention_mask, token_type_ids, input_len,
        token_offsets, text, labels, subjects_ids in tqdm(
            zip(all_input_ids, all_attention_mask, all_token_type_ids,
                all_input_lens, all_token_offsets, texts, all_labels,
                all_subjects_ids),
            desc="build all_features")
    ]

    return all_features
