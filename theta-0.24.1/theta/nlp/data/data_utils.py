#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
from loguru import logger

from datasets.arrow_dataset import Dataset as ArrowDataset
from datasets.dataset_dict import DatasetDict


def yield_data_generator(iterable):
    def _generator():
        for x in tqdm(iterable):
            yield x

    return _generator


def check_labels(special_labels, labels_list):
    unknown_labels = set(labels_list) - set(special_labels)
    if any(unknown_labels):
        logger.warning(
            f"Labels {unknown_labels} not in special_labels: {special_labels}")
    notexist_labels = set(special_labels) - set(labels_list)
    if any(notexist_labels):
        logger.warning(
            f"Labels {notexist_labels} not in labels_list: {set(labels_list)}")


def convert_glue_samples_to_hf_dataset(samples, glue_labels, is_test=False):
    idx_list = []
    text_a_list = []
    text_b_list = []
    labels_list = []

    for guid, text_a, text_b, label in tqdm(samples):
        idx_list.append(guid)
        text_a_list.append(text_a)
        text_b_list.append(text_b)
        if not is_test:
            labels_list.append(label)

    data_dict = {
        'idx': idx_list,
        'sentence1': text_a_list,
    }
    if any(text_b_list):
        data_dict['sentence2'] = text_b_list
    if not is_test:
        check_labels(glue_labels, labels_list)
        data_dict['label'] = labels_list

    dataset = ArrowDataset.from_dict(data_dict)

    return dataset


def encode_glue_samples_to_hf_dataset(samples,
                                      encode_datasets_fn,
                                      glue_labels,
                                      tokenizer,
                                      data_args,
                                      is_test=False):
    arrow_dataset = convert_glue_samples_to_hf_dataset(samples,
                                                       glue_labels,
                                                       is_test=is_test)

    hf_datasets = DatasetDict({
        'dataset': arrow_dataset,
    })

    label_to_id = {v: i for i, v in enumerate(glue_labels)}
    sentence1_key = "sentence1"
    sentence2_key = "sentence2"

    hf_datasets = encode_datasets_fn(tokenizer, hf_datasets, data_args,
                                     label_to_id, sentence1_key, sentence2_key)

    return hf_datasets['dataset']
