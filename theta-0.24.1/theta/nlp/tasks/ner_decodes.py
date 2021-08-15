#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
from loguru import logger
import numpy as np
from collections import defaultdict


def crf_decode(decode_tokens, raw_text, id2ent):
    """
    CRF 解码，用于解码 time loc 的提取
    """
    predict_entities = {}

    decode_tokens = decode_tokens[1:-1]  # 除去 CLS SEP token

    index_ = 0

    while index_ < len(decode_tokens):

        token_label = id2ent[decode_tokens[index_]].split('-')

        if token_label[0].startswith('S'):
            token_type = token_label[1]
            tmp_ent = raw_text[index_]

            if token_type not in predict_entities:
                predict_entities[token_type] = [(tmp_ent, index_)]
            else:
                predict_entities[token_type].append((tmp_ent, int(index_)))

            index_ += 1

        elif token_label[0].startswith('B'):
            token_type = token_label[1]
            start_index = index_

            index_ += 1
            while index_ < len(decode_tokens):
                temp_token_label = id2ent[decode_tokens[index_]].split('-')

                if temp_token_label[0].startswith(
                        'I') and token_type == temp_token_label[1]:
                    index_ += 1
                elif temp_token_label[0].startswith(
                        'E') and token_type == temp_token_label[1]:
                    end_index = index_
                    index_ += 1

                    tmp_ent = raw_text[start_index:end_index + 1]

                    if token_type not in predict_entities:
                        predict_entities[token_type] = [(tmp_ent, start_index)]
                    else:
                        predict_entities[token_type].append(
                            (tmp_ent, int(start_index)))

                    break
                else:
                    break
        else:
            index_ += 1

    return predict_entities


# 严格解码 baseline
def span_decode(start_logits, end_logits, raw_text, id2ent):
    predict_entities = defaultdict(list)

    start_pred = np.argmax(start_logits, -1)
    end_pred = np.argmax(end_logits, -1)

    for i, s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j, e_type in enumerate(end_pred[i:]):
            if s_type == e_type:
                tmp_ent = raw_text[i:i + j + 1]
                predict_entities[id2ent[s_type]].append((tmp_ent, i))
                break

    return predict_entities


# 严格解码 baseline
def mrc_decode(start_logits, end_logits, raw_text):
    predict_entities = []
    start_pred = np.argmax(start_logits, -1)
    end_pred = np.argmax(end_logits, -1)

    for i, s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j, e_type in enumerate(end_pred[i:]):
            if s_type == e_type:
                tmp_ent = raw_text[i:i + j + 1]
                predict_entities.append((tmp_ent, i))
                break

    return predict_entities
