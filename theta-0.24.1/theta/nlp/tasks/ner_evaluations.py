#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from loguru import logger
import numpy as np
from collections import defaultdict
from .ner_decodes import crf_decode, span_decode, mrc_decode


def calculate_metric(gt, predict):
    """
    计算 tp fp fn
    """
    tp, fp, fn = 0, 0, 0
    for entity_predict in predict:
        flag = 0
        for entity_gt in gt:
            if entity_predict[0] == entity_gt[0] and entity_predict[
                    1] == entity_gt[1]:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1

    fn = len(gt) - tp

    return np.array([tp, fp, fn])


def get_p_r_f(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])


def crf_evaluation(model, dev_info, device, ent2id):
    dev_loader, (dev_callback_info, type_weight) = dev_info

    pred_tokens = []

    for tmp_pred in get_base_out(model, dev_loader, device):
        pred_tokens.extend(tmp_pred[0])

    assert len(pred_tokens) == len(dev_callback_info)

    id2ent = {ent2id[key]: key for key in ent2id.keys()}

    role_metric = np.zeros([13, 3])

    mirco_metrics = np.zeros(3)

    for tmp_tokens, tmp_callback in zip(pred_tokens, dev_callback_info):

        text, gt_entities = tmp_callback

        tmp_metric = np.zeros([13, 3])

        pred_entities = crf_decode(tmp_tokens, text, id2ent)

        for idx, _type in enumerate(ENTITY_TYPES):
            if _type not in pred_entities:
                pred_entities[_type] = []

            tmp_metric[idx] += calculate_metric(gt_entities[_type],
                                                pred_entities[_type])

        role_metric += tmp_metric

    for idx, _type in enumerate(ENTITY_TYPES):
        temp_metric = get_p_r_f(role_metric[idx][0], role_metric[idx][1],
                                role_metric[idx][2])

        mirco_metrics += temp_metric * type_weight[_type]

    metric_str = f'[MIRCO] precision: {mirco_metrics[0]:.4f}, ' \
                 f'recall: {mirco_metrics[1]:.4f}, f1: {mirco_metrics[2]:.4f}'

    return metric_str, mirco_metrics[2]


def span_evaluation(model, dev_info, device, ent2id):
    dev_loader, (dev_callback_info, type_weight) = dev_info

    start_logits, end_logits = None, None

    model.eval()

    for tmp_pred in get_base_out(model, dev_loader, device):
        tmp_start_logits = tmp_pred[0].cpu().numpy()
        tmp_end_logits = tmp_pred[1].cpu().numpy()

        if start_logits is None:
            start_logits = tmp_start_logits
            end_logits = tmp_end_logits
        else:
            start_logits = np.append(start_logits, tmp_start_logits, axis=0)
            end_logits = np.append(end_logits, tmp_end_logits, axis=0)

    assert len(start_logits) == len(end_logits) == len(dev_callback_info)

    role_metric = np.zeros([13, 3])

    mirco_metrics = np.zeros(3)

    id2ent = {ent2id[key]: key for key in ent2id.keys()}

    for tmp_start_logits, tmp_end_logits, tmp_callback \
            in zip(start_logits, end_logits, dev_callback_info):

        text, gt_entities = tmp_callback

        tmp_start_logits = tmp_start_logits[1:1 + len(text)]
        tmp_end_logits = tmp_end_logits[1:1 + len(text)]

        pred_entities = span_decode(tmp_start_logits, tmp_end_logits, text,
                                    id2ent)

        for idx, _type in enumerate(ENTITY_TYPES):
            if _type not in pred_entities:
                pred_entities[_type] = []

            role_metric[idx] += calculate_metric(gt_entities[_type],
                                                 pred_entities[_type])

    for idx, _type in enumerate(ENTITY_TYPES):
        temp_metric = get_p_r_f(role_metric[idx][0], role_metric[idx][1],
                                role_metric[idx][2])

        mirco_metrics += temp_metric * type_weight[_type]

    metric_str = f'[MIRCO] precision: {mirco_metrics[0]:.4f}, ' \
                 f'recall: {mirco_metrics[1]:.4f}, f1: {mirco_metrics[2]:.4f}'

    return metric_str, mirco_metrics[2]


def mrc_evaluation(model, dev_info, device):
    dev_loader, (dev_callback_info, type_weight) = dev_info

    start_logits, end_logits = None, None

    model.eval()

    for tmp_pred in get_base_out(model, dev_loader, device):
        tmp_start_logits = tmp_pred[0].cpu().numpy()
        tmp_end_logits = tmp_pred[1].cpu().numpy()

        if start_logits is None:
            start_logits = tmp_start_logits
            end_logits = tmp_end_logits
        else:
            start_logits = np.append(start_logits, tmp_start_logits, axis=0)
            end_logits = np.append(end_logits, tmp_end_logits, axis=0)

    assert len(start_logits) == len(end_logits) == len(dev_callback_info)

    role_metric = np.zeros([13, 3])

    mirco_metrics = np.zeros(3)

    id2ent = {x: i for i, x in enumerate(ENTITY_TYPES)}

    for tmp_start_logits, tmp_end_logits, tmp_callback \
            in zip(start_logits, end_logits, dev_callback_info):

        text, text_offset, ent_type, gt_entities = tmp_callback

        tmp_start_logits = tmp_start_logits[text_offset:text_offset +
                                            len(text)]
        tmp_end_logits = tmp_end_logits[text_offset:text_offset + len(text)]

        pred_entities = mrc_decode(tmp_start_logits, tmp_end_logits, text)

        role_metric[id2ent[ent_type]] += calculate_metric(
            gt_entities, pred_entities)

    for idx, _type in enumerate(ENTITY_TYPES):
        temp_metric = get_p_r_f(role_metric[idx][0], role_metric[idx][1],
                                role_metric[idx][2])

        mirco_metrics += temp_metric * type_weight[_type]

    metric_str = f'[MIRCO] precision: {mirco_metrics[0]:.4f}, ' \
                  f'recall: {mirco_metrics[1]:.4f}, f1: {mirco_metrics[2]:.4f}'

    return metric_str, mirco_metrics[2]
