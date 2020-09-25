#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
from loguru import logger
import pandas as pd

train_file = "data/rawdata/ccks_7_1_competition_data/entity_type.txt"
test_file = "data/rawdata/ccks_7_1_competition_data/entity_validation.txt",
#  submission_file = "submissions/entity_typing_submission_6da7de8eb3a611eabda9fa163e51b5c3.txt"
submission_file = "submissions/entity_typing_submission_e21bb4b4bc3611eaace7fa163e51b5c3.txt"

data_file = train_file
#  data_file = submission_file

df_train = pd.read_csv(data_file,
                       sep='\t',
                       header=None,
                       names=['text', 'label'])
df_symptom = df_train[df_train.label == '症状']
df_disease = df_train[df_train.label == '疾病']

df_data = pd.concat([df_symptom, df_disease], axis=0)

lastchar_categories = {}
lastchar2_categories = {}

for i, row in tqdm(df_data.iterrows()):
    lastchar = row.text[-1]
    lastchar2 = row.text[-2:]

    if lastchar not in lastchar_categories:
        lastchar_categories[lastchar] = (0, 0, 0)
    n, a, b = lastchar_categories[lastchar]
    if row.label == '症状':
        lastchar_categories[lastchar] = (n + 1, a + 1, b)
    else:
        lastchar_categories[lastchar] = (n + 1, a, b + 1)

    if lastchar2 not in lastchar2_categories:
        lastchar2_categories[lastchar2] = (0, 0, 0)
    n, a, b = lastchar2_categories[lastchar2]
    if row.label == '症状':
        lastchar2_categories[lastchar2] = (n + 1, a + 1, b)
    else:
        lastchar2_categories[lastchar2] = (n + 1, a, b + 1)

lastchar_categories = sorted(lastchar_categories.items(),
                             key=lambda x: x[1][0],
                             reverse=True)
lastchar2_categories = sorted(lastchar2_categories.items(),
                              key=lambda x: x[1][0],
                              reverse=True)

#  lastchar_categories = [x for x in lastchar_categories if x[1][0] >= 2]
#  lastchar2_categories = [x for x in lastchar2_categories if x[1][0] >= 2]

#  lastchar_categories = [
#      x for x in lastchar_categories
#      if x[1][0] >= 5 and (x[1][1] <= 10 or x[1][2] <= 10)
#  ]
#  lastchar2_categories = [
#      x for x in lastchar2_categories
#      if x[1][0] >= 5 and (x[1][1] <= 10 or x[1][2] <= 10)
#  ]

logger.info(f"{lastchar_categories}")
logger.info(f"{lastchar2_categories}")
