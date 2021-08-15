#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PointerSequenceTagger

指针序列标注器

"""

from copy import deepcopy
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from .base_tagger import BaseTagger
from .tagger_utils import batch_encode_texts, batch_encode_tags, common_to_tensors


class InputFeature(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask, token_type_ids, input_len,
                 token_offsets, start_ids, end_ids, encoded_tags, text):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.token_offsets = token_offsets

        self.start_ids = start_ids
        self.end_ids = end_ids
        self.encoded_tags = encoded_tags
        self.text = text

    def __repr__(self):
        return json.dumps(self.to_dict(),
                          ensure_asci=False,
                          indent=2,
                          sort_keys=True)

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        return deepcopy(self.__dict__)


def encode_examples(examples, tokenizer, label2id, max_seq_length):
    """
    :param 
    """

    # ------------------------------ encode all texts ------------------------------
    all_texts = [e.text for e in examples]
    all_tokens, all_token2char, all_char2token, all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets = batch_encode_texts(
        all_texts, label2id, tokenizer, max_seq_length)

    # ------------------------------ encode all tags ------------------------------
    all_tags = [e.tags for e in examples]
    all_token_lens = [len(input_ids) for input_ids in all_input_ids]
    all_start_ids, all_end_ids, all_encoded_tags = batch_encode_tags(
        all_tags, all_token_lens, all_char2token, label2id, max_seq_length)

    # ------------------------------ check ------------------------------
    logger.debug(f"all_input_ids.shape: {np.array(all_input_ids).shape}")
    logger.debug(
        f"all_attention_mask.shape: {np.array(all_attention_mask).shape}")
    logger.debug(
        f"all_token_type_ids.shape: {np.array(all_token_type_ids).shape}")
    logger.debug(f"all_start_ids.shape: {np.array(all_start_ids).shape}")
    logger.debug(f"all_end_ids.shape: {np.array(all_end_ids).shape}")
    logger.debug(f"all_encoded_tags.shape: {np.array(all_encoded_tags).shape}")
    logger.debug(f"all_input_lens.shape: {np.array(all_input_lens).shape}")
    assert np.array(all_input_ids).shape[1] == max_seq_length
    assert np.array(all_attention_mask).shape[1] == max_seq_length
    assert np.array(all_token_type_ids).shape[1] == max_seq_length

    # ------------------------------ convert to tensor ------------------------------
    all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets = common_to_tensors(
        all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens,
        all_token_offsets)

    all_start_ids = torch.from_numpy(np.array(all_start_ids, dtype=np.int64))
    all_end_ids = torch.from_numpy(np.array(all_end_ids, dtype=np.int64))

    # ------------------------------ build features ------------------------------
    all_features = [
        InputFeature(input_ids=input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids,
                     input_len=input_len,
                     token_offsets=token_offsets,
                     start_ids=start_ids,
                     end_ids=end_ids,
                     encoded_tags=encoded_tags,
                     text=text)
        for input_ids, attention_mask, token_type_ids, start_ids, end_ids,
        encoded_tags, input_len, token_offsets, text in zip(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_start_ids, all_end_ids, all_encoded_tags, all_input_lens,
            all_token_offsets, all_texts)
    ]

    return all_features


class PointerSequenceTagger(BaseTagger):
    def __init__(self, tokenizer, label2id):
        super(PointerSequenceTagger, self).__init__(tokenizer, label2id)

    def encode(self, examples, max_seq_length, **kwarg):
        #  tokenizer = kwarg.get('tokenizer', self.tokenizer)
        #  label2id = kwarg.get('label2id', self.label2id)
        tokenizer = self.tokenizer
        label2id = self.label2id
        return encode_examples(examples, tokenizer, label2id, max_seq_length)
