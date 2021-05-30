# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
from loguru import logger
import pandas as pd
import numpy as np
from collections import Counter

import sys
sys.path.append('../../')

seg_len=0
seg_backoff=0
fold = 0

train_file = './data/rawdata/train.json'
test_file = './data/rawdata/test.json'
dev_file = './data/rawdata/dev.json'

def load_json_data(json_file):
    rd = open(json_file, 'r')
    lines = rd.readlines()
    rd.close()
    json_data = []
    for line in tqdm(lines):
        line = line.strip()
        line_data = json.loads(line)
        json_data.append(line_data)
    print(f"Total: {len(json_data)}")
    print(json_data[:5])
    return json_data

train_data = load_json_data(train_file)
test_data = load_json_data(test_file)
dev_data = load_json_data(dev_file)

print('======================= done ==============================')

all_data = train_data + dev_data
all_labels = []
for text_data in tqdm(all_data):
    labels = text_data['label']
    for k, v in labels.items():
    	all_labels.append(k)
print(f"{Counter(all_labels)}")

import os, sys, json, random
from collections import Counter
from tqdm import tqdm
from loguru import logger
from pathlib import Path

from theta.utils import load_json_file, split_train_eval_examples
from theta.modeling import LabeledText, load_ner_examples, load_ner_labeled_examples, save_ner_preds, show_ner_datainfo

from theta.modeling.ner_span import load_model, NerTrainer, get_args
#from theta.modeling.ner import load_model, NerTrainer, get_args

def clean_text(text):
    if text:
        text = text.strip()
        #  text = re.sub('\t', ' ', text)
    return text


def train_data_generator(train_file):

    lines = load_json_file(train_file)

    for i, x in enumerate(tqdm(lines)):
        guid = str(i)
        text = clean_text(x['text'])
        sl = LabeledText(guid, text)
        entities = []
        classes = x['label'].keys()
        for c in classes:
            c_labels = x['label'][c]
            #  logger.debug(f"c_labels:{c_labels}")
            for label, span in c_labels.items():
                x0, x1 = span[0]
                sl.add_entity(c, x0, x1)
        print("index: ", str(i), ", text: ", text, ", entities: ", sl.entities)
        break

train_data_generator('./data/rawdata/train.json')
