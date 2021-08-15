#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm

from ...utils import match_tokenized_to_untokenized, seg_generator
from ...utils.multiprocesses import (barrier_leader_process,
                                     barrier_member_processes,
                                     is_multi_processes)
from ..spo_utils import InputExample


class InputFeature(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len, segment_ids,
                 sub_labels, seed_sub, obj_labels, token_offsets, text,
                 spo_list):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_len = input_len
        self.sub_labels = sub_labels
        self.seed_sub = seed_sub
        self.obj_labels = obj_labels
        self.token_offsets = token_offsets
        self.text = text
        self.spo_list = spo_list

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
            self.input_len == other.input_len and \
            self.token_offsets == other.token_offsets

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def batch_to_input_data(batch):
    all_input_ids = torch.stack([e.input_ids for e in batch])
    all_input_mask = torch.stack([e.input_mask for e in batch])
    all_segment_ids = torch.stack([e.segment_ids for e in batch])
    all_sub_labels = torch.stack([e.sub_labels for e in batch])
    all_seed_subs = torch.stack([e.seed_sub for e in batch])
    all_obj_labels = torch.stack([e.obj_labels for e in batch])
    all_lens = torch.stack([e.input_len for e in batch])
    all_token_offsets = torch.stack([e.token_offsets for e in batch])
    all_texts = [e.text_a for e in batch]
    all_spo_lists = [e.spo_list for e in batch]

    return (all_input_ids, all_input_mask, all_segment_ids, all_sub_labels,
            all_seed_subs, all_obj_labels, all_lens, all_token_offsets,
            all_texts, all_spo_lists)


def encode_examples(examples, label2id, tokenizer, max_seq_length):

    num_labels = len(label2id)

    #  texts = [e.text for e in examples]
    #  all_tokens = [['[CLS]'] + tokenizer.encode(
    #      text, add_special_tokens=False).tokens[:max_seq_length - 2] +
    #                ['[SEP]'] for text in tqdm(texts, desc="Tokenize")]
    #  all_token2texts = [
    #      match_tokenized_to_untokenized(tokens, text)
    #      for tokens, text in zip(all_tokens, texts)
    #  ]
    #  all_input_ids = [
    #      tokenizer.convert_tokens_to_ids(tokens) for tokens in all_tokens
    #  ]
    #  all_attention_mask = [[1] * len(input_ids) for input_ids in all_input_ids]
    #  all_token_type_ids = [[0] * len(input_ids) for input_ids in all_input_ids]

    for e in examples:
        assert len(e.text_a) <= max_seq_length - 2
        #  logger.warning(f"text: {e.text}")
        #  logger.warning(f"spo_list: {e.spo_list}")
    texts = [e.text_a[:max_seq_length - 2] for e in examples]
    all_encodes = [
        tokenizer.encode(text, add_special_tokens=True)
        for text in tqdm(texts, desc="Tokenize")
    ]

    #  all_tokens = [encode.tokens for encode in all_encodes]
    #  all_token_offsets = [encode.offsets for encode in all_encodes]
    #  all_input_ids = [encode.ids for encode in all_encodes]
    #  all_attention_mask = [encode.attention_mask for encode in all_encodes]
    #  all_token_type_ids = [encode.type_ids for encode in all_encodes]

    all_encodes = tokenizer.batch_encode(texts, add_special_tokens=True)
    all_tokens = all_encodes['tokens']
    all_token_offsets = all_encodes['offsets']
    all_token2char = all_encodes['token2char']
    all_char2token = all_encodes['char2token']

    all_input_ids = all_encodes['ids']
    all_attention_mask = all_encodes['attention_mask']
    all_token_type_ids = all_encodes['type_ids']

    all_input_lens = [len(tokens) for tokens in all_tokens]

    all_padding_lens = [max_seq_length - n for n in all_input_lens]
    for i, (input_ids, attention_mask, token_type_ids, token2char,
            token_offsets, padding_length) in enumerate(
                zip(all_input_ids, all_attention_mask, all_token_type_ids,
                    all_token2char, all_token_offsets, all_padding_lens)):
        all_input_ids[i] = input_ids + [0] * padding_length
        all_attention_mask[i] = attention_mask + [0] * padding_length
        all_token_type_ids[i] = token_type_ids + [0] * padding_length
        all_token2char[i] = token2char + [0] * padding_length
        all_token_offsets[i] = token_offsets + [(0, 0)] * padding_length

    def encode_spo(spo_list, num_tokens, text_len, char2token):
        """
        spo_list: [(s, p, o), ...]
            s: (sub_s, sub_e)
            p: predicate
            o: (obj_s, obj_e)
        #  spo_list: [spo1, spo2, ...]
        #      spo: {  'subject':   {'category': sub_c, 'start': sub_s, 'mention': sub_m},
        #              'predicate': predicate,
        #              'object':    {'category': obj_c, 'start': obj_s, 'mention': obj_m}
        #          }

        """
        # 将spo三元组列表转换为词典{(s0, s1): [(o_0, o_1, p)]}
        spoes = defaultdict(list)
        if spo_list:
            for s, p, o in spo_list:
                s_start, s_mention = s
                s_end = s_start + len(s_mention) - 1
                #  assert s_end <= text_len, f"Skip spo. s_end >= {text_len}: {s}, {p}, {o}. "

                o_start, o_mention = o
                o_end = o_start + len(o_mention) - 1
                #  assert o_end <= text_len, f"Skip spo. o_end >= {text_len}: {s}, {p}, {o}. "

                #  logger.warning(f"s: {s_start}:{s_end}, o: {o_start}:{o_end}")
                #  logger.info(f"char2toke: {char2token}")
                s_start = char2token[s_start]
                assert s_start >= 0 and s_start < num_tokens - 2, f"s_start overflow. |{s}|{p}|{o}| ({s_start}:{s_end}, num_tokens: {num_tokens})"
                s_end = char2token[s_end]
                assert s_end >= 0 and s_end < num_tokens - 2, f"s_end overflow. |{s}|{p}|{o}| ({s_start}:{s_end}, num_tokens: {num_tokens})"
                o_start = char2token[o_start]
                assert o_start >= 0 and o_start < num_tokens - 2, f"o_start overflow. |{s}|{p}|{o}| ({o_start}:{o_end}, num_tokens: {num_tokens})"
                o_end = char2token[o_end]
                assert o_end >= 0 and o_end < num_tokens - 2, f"o_end overflow. |{s}|{p}|{o}| ({o_start}:{o_end}, num_tokens: {num_tokens})"

                spoes[(s_start, s_end)].append((o_start, o_end, label2id[p]))

        sub_labels = np.zeros((num_tokens, 2), dtype=int)
        # 对应的object标签
        obj_labels = np.zeros((num_tokens, num_labels, 2), dtype=int)
        #  logger.info(f"obj_labels.shape: {obj_labels.shape}")
        seed_sub = (0, 0)
        if spoes:

            for s in spoes:
                sub_labels[s[0] + 1, 0] = 1
                sub_labels[s[1] + 1, 1] = 1

            #  logger.info(f"spoes: {spoes}")
            # 随机选一个subject
            start, end = np.array(list(spoes.keys())).T
            start = np.random.choice(start)
            end = np.random.choice(end[end >= start])
            seed_sub = (start, end)

            for o in spoes.get(seed_sub, []):
                #  logger.info(f"o: {o}")
                obj_labels[o[0] + 1, o[2], 0] = 1
                obj_labels[o[1] + 1, o[2], 1] = 1

        return sub_labels, seed_sub, obj_labels

    all_sub_labels = []
    all_seed_subs = []
    all_obj_labels = []

    all_spo_lists = [e.spo_list for e in examples]
    for example, input_ids, attention_mask, token_type_ids, spo_list, input_len, char2token in tqdm(
            zip(examples, all_input_ids, all_attention_mask,
                all_token_type_ids, all_spo_lists, all_input_lens,
                all_char2token)):
        num_tokens = len(input_ids)
        #  logger.info(f"example.text: {example.text}")
        sub_labels, seed_sub, obj_labels = encode_spo(spo_list, num_tokens,
                                                      input_len - 2,
                                                      char2token)
        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(sub_labels) == max_seq_length
        assert len(obj_labels) == max_seq_length

        all_sub_labels.append(sub_labels)
        all_seed_subs.append(seed_sub)
        all_obj_labels.append(obj_labels)
        #  logger.info(
        #      f"obj_labels: {torch.tensor(obj_labels, dtype=torch.long).cuda()}")

    logger.debug(f"all_input_ids.shape: {np.array(all_input_ids).shape}")
    logger.debug(
        f"all_attention_mask.shape: {np.array(all_attention_mask).shape}")
    logger.debug(
        f"all_token_type_ids.shape: {np.array(all_token_type_ids).shape}")
    logger.debug(f"all_sub_labels.shape: {np.array(all_sub_labels).shape}")
    logger.debug(f"all_obj_labels.shape: {np.array(all_obj_labels).shape}")
    logger.debug(f"all_input_lens.shape: {np.array(all_input_lens).shape}")
    logger.debug(
        f"all_token_offsets.shape: {np.array(all_token_offsets).shape}")
    assert np.array(all_input_ids).shape[
        1] == max_seq_length, f"shape: {np.array(all_input_ids).shape}"
    assert np.array(all_attention_mask).shape[1] == max_seq_length
    assert np.array(all_token_type_ids).shape[1] == max_seq_length

    for (start, end), text_len in zip(all_seed_subs, all_input_lens):
        assert start < text_len and end < text_len, f"start: {start}, end: {end}, text_len: {text_len}"

    all_input_ids = torch.from_numpy(np.array(all_input_ids, dtype=np.int64))
    all_attention_mask = torch.from_numpy(
        np.array(all_attention_mask, dtype=np.int64))
    all_token_type_ids = torch.from_numpy(
        np.array(all_token_type_ids, dtype=np.int64))
    all_sub_labels = torch.from_numpy(np.array(all_sub_labels, dtype=np.int64))
    all_seed_subs = torch.from_numpy(np.array(all_seed_subs, dtype=np.int64))
    all_obj_labels = torch.from_numpy(np.array(all_obj_labels, dtype=np.int64))
    all_input_lens = torch.from_numpy(np.array(all_input_lens, dtype=np.int64))
    all_token_offsets = torch.from_numpy(
        np.array(all_token_offsets, dtype=np.int64))

    all_features = [
        InputFeature(input_ids=input_ids,
                     input_mask=attention_mask,
                     segment_ids=token_type_ids,
                     sub_labels=sub_labels,
                     seed_sub=seed_subs,
                     obj_labels=obj_labels,
                     input_len=input_len,
                     token_offsets=token_offsets,
                     text=text,
                     spo_list=spo_list)
        for input_ids, attention_mask, token_type_ids, sub_labels, seed_subs,
        obj_labels, input_len, token_offsets, text, spo_list in tqdm(
            zip(all_input_ids, all_attention_mask, all_token_type_ids,
                all_sub_labels, all_seed_subs, all_obj_labels, all_input_lens,
                all_token_offsets, texts, all_spo_lists))
    ]
    logger.warning(f"Prepare data done.")
    #  logger.info(f"all_token_offsets[:1]: {all_token_offsets[:1]}")
    #  logger.info(f"{np.array(all_token_offsets).shape}")
    #  logger.info(f"{np.array(all_token_offsets[:1]).shape}")
    #  logger.info(f"{np.array(all_token_offsets[:2]).shape}")

    return all_features

    #  return (all_features, {
    #      'input_ids': all_input_ids,
    #      'attention_mask': all_attention_mask,
    #      'token_type_ids': all_token_type_ids,
    #      'sub_labels': all_sub_labels,
    #      "seed_sub": all_seed_subs,
    #      'obj_labels': all_obj_labels,
    #      "input_lens": all_input_lens,
    #      "token_offsets": all_token_offsets,
    #  })

    #  return (all_features, {
    #  'input_ids':
    #  torch.from_numpy(np.array(all_input_ids, dtype=np.int64)),
    #  'attention_mask':
    #  torch.from_numpy(np.array(all_attention_mask, dtype=np.int64)),
    #  'token_type_ids':
    #  torch.from_numpy(np.array(all_token_type_ids, dtype=np.int64)),
    #  'sub_labels':
    #  torch.from_numpy(np.array(all_sub_labels, dtype=np.int64)),
    #  "seed_sub":
    #  torch.from_numpy(np.array(all_seed_subs, dtype=np.int64)),
    #  'obj_labels':
    #  torch.from_numpy(np.array(all_obj_labels, dtype=np.int64)),
    #  "input_lens":
    #  torch.from_numpy(np.array(all_input_lens, dtype=np.int64)),
    #  "token_offsets":
    #  torch.from_numpy(np.array(all_token_offsets, dtype=np.int64)),
    #  })


#

#  def examples_to_dataset(examples, label2id, tokenizer, max_seq_length):
#      all_features, outputs = encode_examples(examples, label2id, tokenizer,
#                                              max_seq_length)
#
#      logger.warning(f"Buid TensorDataset")
#      dataset = TensorDataset(
#          outputs['input_ids'],
#          outputs['attention_mask'],
#          outputs['token_type_ids'],
#          outputs['sub_labels'],
#          outputs['seed_sub'],
#          outputs['obj_labels'],
#          outputs['input_lens'],
#          outputs['token_offsets'],
#      )
#
#      assert len(dataset) == len(all_features)
#
#      return dataset, all_features
