#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from loguru import logger
import torch


def batch_encode_texts(all_texts, label2id, tokenizer, max_seq_length):
    all_encodes = tokenizer.batch_encode(all_texts, add_special_tokens=True)
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
                tqdm(zip(all_input_ids, all_attention_mask, all_token_type_ids,
                         all_token2char, all_token_offsets, all_padding_lens),
                     desc="common_batch_encode")):

        if padding_length < 0:
            all_tokens[i] = all_tokens[i][:max_seq_length - 1] + ['[SEP]']
            all_input_ids[i] = input_ids[:max_seq_length - 1] + [102]
            all_attention_mask[i] = attention_mask[:max_seq_length]
            all_token_type_ids[i] = token_type_ids[:max_seq_length]
            all_token2char[i] = token2char[:max_seq_length]
            all_token_offsets[i] = token_offsets[:max_seq_length - 1] + [(0, 0)
                                                                         ]
            #  logger.warning(f"all_tokens[i]: {all_tokens[i]}")
            #  logger.debug(f"all_input_ids[i]: {all_input_ids[i]}")
            #  logger.debug(f"all_token_offsets[i]: {all_token_offsets[i]}")

        else:
            all_input_ids[i] = input_ids + [0] * padding_length
            all_attention_mask[i] = attention_mask + [0] * padding_length
            all_token_type_ids[i] = token_type_ids + [0] * padding_length
            all_token2char[i] = token2char + [0] * padding_length
            all_token_offsets[i] = token_offsets + [(0, 0)] * padding_length
        all_input_lens[i] = len(input_ids)

        #  logger.debug(
        #      f"padding_length: {padding_length}, len(input_ids): {len(input_ids)}"
        #  )
        assert len(all_input_ids[i]) == max_seq_length
        assert len(all_attention_mask[i]) == max_seq_length
        assert len(all_token_type_ids[i]) == max_seq_length
        assert len(all_token2char[i]) == max_seq_length
        assert len(all_token_offsets[i]) == max_seq_length

    #  logger.warning(f"all_input_ids: {all_input_ids}")
    all_input_ids = torch.from_numpy(np.array(all_input_ids, dtype=np.int64))
    all_attention_mask = torch.from_numpy(
        np.array(all_attention_mask, dtype=np.int64))
    all_token_type_ids = torch.from_numpy(
        np.array(all_token_type_ids, dtype=np.int64))
    all_input_lens = torch.from_numpy(np.array(all_input_lens, dtype=np.int64))
    all_token_offsets = torch.from_numpy(
        np.array(all_token_offsets, dtype=np.int64))

    return (all_tokens, all_token2char, all_char2token, all_input_ids,
            all_attention_mask, all_token_type_ids, all_input_lens,
            all_token_offsets)


def batch_encode_tags(all_tags, all_token_lens, all_char2token, label2id,
                      max_seq_length):
    def encode_tags(tags, num_tokens, char2token):
        #  logger.warning(f"num_tokens: {num_tokens}")
        start_ids = [label2id['[unused1]']] * num_tokens
        end_ids = [label2id['[unused1]']] * num_tokens

        encoded_tags = []
        for tag in tags:
            label = tag.category
            start = tag.start
            end = tag.start + len(tag.mention)

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
            encoded_tags.append((label2id[label], start, end))

        return encoded_tags, start_ids, end_ids

    all_encoded_tags = []
    all_start_ids = []
    all_end_ids = []

    for tags, num_tokens, char2token in zip(all_tags, all_token_lens,
                                            all_char2token):

        encoded_tags, start_ids, end_ids = encode_tags(tags, num_tokens,
                                                       char2token)

        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length

        all_encoded_tags.append(encoded_tags)
        all_start_ids.append(start_ids)
        all_end_ids.append(end_ids)

    return all_start_ids, all_end_ids, all_encoded_tags


def common_to_tensors(all_input_ids, all_attention_mask, all_token_type_ids,
                      all_input_lens, all_token_offsets):
    all_input_ids = torch.from_numpy(np.array(all_input_ids, dtype=np.int64))
    all_attention_mask = torch.from_numpy(
        np.array(all_attention_mask, dtype=np.int64))
    all_token_type_ids = torch.from_numpy(
        np.array(all_token_type_ids, dtype=np.int64))
    all_input_lens = torch.from_numpy(np.array(all_input_lens, dtype=np.int64))
    all_token_offsets = torch.from_numpy(
        np.array(all_token_offsets, dtype=np.int64))

    return all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets
