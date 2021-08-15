#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
from datasets.arrow_dataset import Dataset as ArrowDataset


def convert_glue_samples_to_dataset(samples, is_test=False):
    idx_list = []
    text_a_list = []
    text_b_list = []
    label_list = []

    for guid, text_a, text_b, label in tqdm(samples):
        idx_list.append(guid)
        text_a_list.append(text_a)
        text_b_list.append(text_b)
        if not is_test:
            label_list.append(label)

    data_dict = {
        'idx': idx_list,
        'sentence1': text_a_list,
    }
    if any(text_b_list):
        data_dict['sentence2'] = text_b_list
    if not is_test:
        data_dict['label'] = label_list

    dataset = ArrowDataset.from_dict(data_dict)

    return dataset
