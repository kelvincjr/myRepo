#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json

import numpy as np
from loguru import logger
from tqdm import tqdm
import rich
from collections import defaultdict


def get_module(module_name):
    import importlib
    module_cls = importlib.import_module(module_name)

    return module_cls


def main():
    import sys
    module_name = sys.argv[1]
    module_cls = get_module(module_name)

    train_data_generator = None
    if 'train_data_generator' in module_cls.__dict__:
        train_data_generator = module_cls.train_data_generator
    test_data_generator = None
    if 'test_data_generator' in module_cls.__dict__:
        test_data_generator = module_cls.test_data_generator

    is_ner = False
    is_glue = False
    if 'ner_labels' in module_cls.__dict__:
        is_ner = True
    if 'glue_labels' in module_cls.__dict__:
        is_glue = True

    if train_data_generator is not None:
        if is_glue:
            train_samples = [x for x in train_data_generator()]
            labels = [label for _, _, _, label in train_samples]
            num_train_samples = len(labels)

            text1_lens = [len(text_a) for _, text_a, _, _ in train_samples]
            len_mean1 = np.mean(text1_lens)
            len_std1 = np.std(text1_lens)
            len_min1 = np.min(text1_lens)
            len_max1 = np.max(text1_lens)

            text2_lens = [
                len(text_b) if text_b is not None else 0
                for _, _, text_b, _ in train_samples
            ]
            len_mean2 = np.mean(text2_lens)
            len_std2 = np.std(text2_lens)
            len_min2 = np.min(text2_lens)
            len_max2 = np.max(text2_lens)

            labels = sorted(list(set(labels)))
            rich.print(f"{len(labels)} glue_labels: {labels}")
            rich.print(f"num_train_samples: {num_train_samples}")
            rich.print(f"text_a lengths:")
            rich.print(
                f"mean: {len_mean1:.2f}, std: {len_std1:.2f}, min: {len_min1}, max: {len_max1}"
            )
            rich.print(f"text_b lengths:")
            rich.print(
                f"mean: {len_mean2:.2f}, std: {len_std2:.2f}, min: {len_min2}, max: {len_max2}"
            )
            # 标签分布情况
            labels_map = defaultdict(int)
            for sample in train_samples:
                c = sample[3]
                labels_map[c] += 1
            rich.print(f"{labels_map}")
        elif is_ner:
            train_samples = [x for x in train_data_generator()]
            all_tags = [tags for _, _, _, tags in train_samples]
            num_train_samples = len(all_tags)

            labels = []
            for tags in all_tags:
                for tag in tags:
                    c = tag['category']
                    if c not in labels:
                        labels.append(c)

            text_lens = [len(text) for _, text, _, _ in train_samples]
            len_mean = np.mean(text_lens)
            len_std = np.std(text_lens)
            len_min = np.min(text_lens)
            len_max = np.max(text_lens)

            labels = sorted(list(set(labels)))
            rich.print(f"{len(labels)} ner labels: {labels}")
            rich.print(f"num_train_samples: {num_train_samples}")
            rich.print(f"text lengths:")
            rich.print(
                f"mean: {len_mean:.2f}, std: {len_std:.2f}, min: {len_min}, max: {len_max}"
            )

    if test_data_generator is not None:
        if is_glue:
            test_samples = [x for x in test_data_generator()]
            text1_lens = [len(text_a) for _, text_a, _, _ in test_samples]
            len_mean1 = np.mean(text1_lens)
            len_std1 = np.std(text1_lens)
            len_min1 = np.min(text1_lens)
            len_max1 = np.max(text1_lens)

            text2_lens = [
                len(text_b) if text_b is not None else 0
                for _, _, text_b, _ in test_samples
            ]
            len_mean2 = np.mean(text2_lens)
            len_std2 = np.std(text2_lens)
            len_min2 = np.min(text2_lens)
            len_max2 = np.max(text2_lens)
            num_test_samples = len(test_samples)
            rich.print(f"num_test_samples: {num_test_samples}")
            rich.print(f"text_a lengths:")
            rich.print(
                f"mean: {len_mean1:.2f}, std: {len_std1:.2f}, min: {len_min1}, max: {len_max1}"
            )
            rich.print(f"text_b lengths:")
            rich.print(
                f"mean: {len_mean2:.2f}, std: {len_std2:.2f}, min: {len_min2}, max: {len_max2}"
            )
        elif is_ner:
            test_samples = [x for x in test_data_generator()]
            text_lens = [len(text) for _, text, _, _ in test_samples]
            num_test_samples = len(text_lens)

            len_mean = np.mean(text_lens)
            len_std = np.std(text_lens)
            len_min = np.min(text_lens)
            len_max = np.max(text_lens)

            rich.print(f"num_test_samples: {num_test_samples}")
            rich.print(f"text lengths:")
            rich.print(
                f"mean: {len_mean:.2f}, std: {len_std:.2f}, min: {len_min}, max: {len_max}"
            )


if __name__ == "__main__":
    main()
