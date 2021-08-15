#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, copy
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from ...utils.multiprocesses import barrier_leader_process, barrier_member_processes, is_multi_processes
from ...utils import seg_generator


class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, spo_list, text_offset=0):
        self.guid = guid
        self.text_a = text_a
        self.spo_list = spo_list
        self.text_offset = text_offset

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeature(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len, segment_ids,
                 sub_labels, seed_sub, obj_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_len = input_len
        self.sub_labels = sub_labels
        self.seed_sub = seed_sub
        self.obj_labels = obj_labels

    def __repr__(self):
        return str(self.to_json_string())

    def __eq__(self, other):
        return self.input_ids == other.input_ids and \
            self.input_mask == other.input_mask and \
            self.segment_ids == other.segment_ids and \
            self.labels == other.labels and \
            self.sub_labels == other.sub_labels and \
            self.seed_sub == other.seed_sub and \
            self.obj_labels == other.obj_labels and \
            self.input_len == other.input_len

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def encode_examples(examples, label2id, tokenizer, max_seq_length):

    num_labels = len(label2id)
    texts = [e.text_a for e in examples]

    all_tokens = [[tokenizer.cls_token] +
                  tokenizer.tokenize(text)[:max_seq_length - 2] +
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

    def encode_spo(spo_list, num_tokens):
        """
        spo_list: [(s, p, o), ...]
            s: (sub_s, sub_e)
            p: predicate
            o: (obj_s, obj_e)

        """
        #  logger.warning(f"num_tokens: {num_tokens}")

        if spo_list:
            # 将spo三元组列表转换为词典{s: (p,o)}
            spoes = {s: (label2id[p], o) for s, p, o in spo}

            sub_labels = np.zeros((num_tokens, 2), dtype=int)
            for s in spoes:
                sub_labels[s[0] + 1, 0] = 1
                sub_labels[s[1] + 1, 1] = 1

            # 随机选一个subject
            start, end = np.array(list(spoes.keys())).T
            start = np.random.choice(start)
            end = np.random.choice(end[end >= start])
            seed_sub = (start, end)
            # 对应的object标签
            object_labels = np.zeros((num_tokens, num_labels, 2))
            for o in spoes.get(seed_sub, []):
                object_labels[o[0], o[2], 0] = 1
                object_labels[o[1], o[2], 1] = 1

        return sub_labels, seed_sub, obj_labels

    all_sub_labels = []
    all_seed_subs = []
    all_obj_labels = []

    all_spo_lists = [e.spo_list for e in examples]
    for input_ids, attention_mask, token_type_ids, spo_list in zip(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_spo_lists):
        num_tokens = len(input_ids)
        sub_labels, seed_sub, obj_labels = encode_spo(spo_list, num_tokens)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(sub_labels) == max_seq_length
        assert len(obj_labels) == max_seq_length

        all_sub_labels.append(sub_labels)
        all_seed_subs.append(seed_sub)
        all_obj_labels.append(obj_labels)

    logger.debug(f"all_input_ids.shape: {np.array(all_input_ids).shape}")
    logger.debug(
        f"all_attention_mask.shape: {np.array(all_attention_mask).shape}")
    logger.debug(
        f"all_token_type_ids.shape: {np.array(all_token_type_ids).shape}")
    logger.debug(f"all_sub_labels.shape: {np.array(all_sub_labels).shape}")
    logger.debug(f"all_obj_labels.shape: {np.array(all_obj_labels).shape}")
    logger.debug(f"all_input_lens.shape: {np.array(all_input_lens).shape}")
    logger.debug(f"all_seed_subs.shape: {np.array(all_seed_subs).shape}")
    assert np.array(all_input_ids).shape[1] == max_seq_length
    assert np.array(all_attention_mask).shape[1] == max_seq_length
    assert np.array(all_token_type_ids).shape[1] == max_seq_length

    all_features = [
        InputFeature(input_ids=input_ids,
                     input_mask=attention_mask,
                     segment_ids=token_type_ids,
                     sub_labels=sub_labels,
                     seed_sub=seed_sub,
                     obj_labels=obj_labels,
                     input_len=input_len) for input_ids, attention_mask,
        token_type_ids, sub_labels, seed_sub, obj_labels, input_len in zip(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_sub_labels, all_seed_subs, all_obj_labels, all_input_lens)
    ]
    return (all_features, {
        'input_ids':
        torch.tensor(all_input_ids, dtype=torch.long),
        'attention_mask':
        torch.tensor(all_attention_mask, dtype=torch.long),
        'token_type_ids':
        torch.tensor(all_token_type_ids, dtype=torch.long),
        'sub_labels':
        torch.tensor(all_sub_labels, dtype=torch.long),
        "seed_sub":
        all_seed_subs,
        'obj_labels':
        torch.tensor(all_obj_labels, dtype=torch.long),
        "input_lens":
        torch.tensor(all_input_lens, dtype=torch.long),
    })


def examples_to_dataset(examples, label2id, tokenizer, max_seq_length):
    all_features, outputs = encode_examples(examples, label2id, tokenizer,
                                            max_seq_length)

    dataset = TensorDataset(
        outputs['input_ids'],
        outputs['attention_mask'],
        outputs['token_type_ids'],
        outputs['sub_labels'],
        outputs['seed_sub'],
        outputs['obj_labels'],
        outputs['input_lens'],
    )

    assert len(dataset) == len(all_features)

    return dataset, all_features
