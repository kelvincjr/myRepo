#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob

#  files = glob.glob(f"results/entity_typing_submission_*.txt")
files = glob.glob(f"results/entity_typing_*.txt")
ensemble_results_file = f"./entity_typing_ensemble_{os.path.basename(os.path.abspath('.'))}.txt"

ensemble_results = []

from tqdm import tqdm
from loguru import logger
from collections import Counter

for i, file in enumerate(files):
    logger.info(f"{file}")
    lines = [line.strip() for line in open(file, 'r')]
    for j, line in enumerate(lines):
        text, label = line.split('\t')
        if i == 0:
            ensemble_results.append([text, [label]])
        else:
            ensemble_results[j][1].append(label)

with open(ensemble_results_file, 'w') as wt:
    for text, labels in tqdm(ensemble_results):
        hot_label = Counter(labels).most_common()[0][0]
        wt.write(f"{text}\t{hot_label}\n")
logger.info(f"Saved {ensemble_results_file}")
