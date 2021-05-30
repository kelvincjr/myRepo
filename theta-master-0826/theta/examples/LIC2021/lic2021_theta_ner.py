# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
from loguru import logger
import pandas as pd
import numpy as np
from collections import Counter

import sys
sys.path.append('../../../')

seg_len=0
seg_backoff=0
fold = 0

#train_file = '/opt/kelvin/python/knowledge_graph/ai_contest/lic2021/DuEE-new/DuEE-master/data/DuEE1.0/duee_train.json'
#dev_file = '/opt/kelvin/python/knowledge_graph/ai_contest/lic2021/DuEE-new/DuEE-master/data/DuEE1.0/duee_dev.json'
#test_file = '/opt/kelvin/python/knowledge_graph/ai_contest/lic2021/DuEE-new/DuEE-master/data/DuEE1.0/duee_test1.json'

train_file = r'E:\ai_contest\lic2021\DuEE-master\data\DuEE1.0\duee_train.json'
dev_file = r'E:\ai_contest\lic2021\DuEE-master\data\DuEE1.0\duee_dev.json'
test_file = r'E:\ai_contest\lic2021\DuEE-master\data\DuEE1.0\duee_test1.json'
schema_file = r'E:\ai_contest\lic2021\DuEE-master\data\DuEE1.0\duee_event_schema.json'

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            arguments = {}
            for event in l['event_list']:
                for argument in event['arguments']:
                    key = argument['argument']
                    value = (event['event_type'], argument['role'], argument['argument_start_index'])
                    arguments[key] = value
            D.append((l['text'], arguments))
    return D

def read_baiduee_schema(filename):
    # 读取schema
    with open(filename, encoding='utf-8') as f:
        id2label, label2id, n = {}, {}, 0
        for l in f:
            l = json.loads(l)
            for role in l['role_list']:
                #if l['event_type'].startswith('灾害'):
                key = (l['event_type'], role['role'])
                id2label[n] = key
                label2id[key] = n
                n += 1
        num_labels = len(id2label) * 2 + 1
        return id2label, label2id, num_labels

train_data = load_data(train_file)
valid_data = load_data(dev_file)
id2label, label2id, num_labels = read_baiduee_schema(schema_file)

ner_labels = [
    f"{key[0]}_{key[1]}" for n, key in id2label.items()
]

print('======================= done ==============================')

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

    data = load_data(train_file)

    for i, x in enumerate(tqdm(data)):
        guid = str(i)
        text = clean_text(x[0])
        arguments = x[1]
        sl = LabeledText(guid, text)
        entities = []
        for key, value in arguments.items():
            argument = key
            event_type = value[0]
            role = value[1]
            start_index = int(value[2])
            c = event_type + "_" + role
            x0 = start_index
            x1 = start_index + len(argument) - 1
            sl.add_entity(c, x0, x1)

        #print("index: ", str(i), ", text: ", text, ", entities: ", sl.entities)
        #if i > 10:
            #break
        yield str(i), text, None, sl.entities

def load_train_val_examples(args):
    lines = []
    for guid, text, _, entities in train_data_generator(args.train_file):
        sl = LabeledText(guid, text, entities)
        lines.append({'guid': guid, 'text': text, 'entities': entities})

    allow_overlap = args.allow_overlap
    if args.num_augements > 0:
        allow_overlap = False

    train_base_examples = load_ner_labeled_examples(
        lines,
        ner_labels,
        seg_len=args.seg_len,
        seg_backoff=args.seg_backoff,
        num_augements=args.num_augements,
        allow_overlap=allow_overlap)

    train_examples, val_examples = split_train_eval_examples(
        train_base_examples,
        train_rate=args.train_rate,
        fold=args.fold,
        shuffle=True)

    logger.info(f"Loaded {len(train_examples)} train examples, "
                f"{len(val_examples)} val examples.")
    return train_examples, val_examples


def test_data_generator(test_file):

    data = load_data(test_file)
    for i, x in enumerate(tqdm(data)):
        guid = str(i)
        text_a = clean_text(x[0])

        yield guid, text_a, None, None


def load_test_examples(args):
    test_base_examples = load_ner_examples(test_data_generator,
                                           args.test_file,
                                           seg_len=args.seg_len,
                                           seg_backoff=args.seg_backoff)

    logger.info(f"Loaded {len(test_base_examples)} test examples.")
    return test_base_examples

#train_data_generator(train_file)

from theta.modeling import Params, CommonParams, NerParams, NerAppParams, log_global_params

experiment_params = NerAppParams(
    CommonParams(
        dataset_name="duee1.0",
        experiment_name="lic2021_ner",
        train_file=train_file,
        eval_file=dev_file,
        test_file=test_file,
        learning_rate=2e-5,
        train_max_seq_length=512,
        eval_max_seq_length=512,
        per_gpu_train_batch_size=4,
        per_gpu_eval_batch_size=4,
        per_gpu_predict_batch_size=4,
        #per_gpu_train_batch_size=16,
        #per_gpu_eval_batch_size=16,
        #per_gpu_predict_batch_size=16,
        seg_len=510,
        seg_backoff=128,
        num_train_epochs=10,
        fold=0,
        num_augements=3,
        enable_kd=False,
        enable_sda=False,
        sda_teachers=2,
        loss_type="CrossEntropyLoss",
        #  loss_type='FocalLoss',
        focalloss_gamma=2.0,
        model_type="bert",
        model_path=
        #  "/opt/share/pretrained/pytorch/hfl/chinese-electra-large-discriminator",
        r"E:\ai_contest\ccks\mytheta\myRepo\theta-master-0826\theta\examples\LIC2021\model_rbt3",
        #  "/opt/share/pretrained/pytorch/bert-base-chinese",
        fp16=False,
        best_index="f1",
        random_type="np"),
    NerParams(ner_labels=ner_labels, ner_type='span'))
    #NerParams(ner_labels=ner_labels, ner_type='crf', no_crf_loss=False))

experiment_params.debug()

# -------------------- Trainer --------------------
class AppTrainer(NerTrainer):
    def __init__(self, args, ner_labels):
        super(AppTrainer, self).__init__(args, ner_labels, build_model=None)


def main(args):

    trainer = AppTrainer(args, ner_labels)
    train_examples, val_examples = load_train_val_examples(args)
    trainer.train(args, train_examples, val_examples)

if __name__ == '__main__':

    def add_special_args(parser):
        parser.add_argument("--generate_submission",
                            action="store_true",
                            help="")
        return parser

    args = get_args(experiment_params=experiment_params,
                    special_args=[add_special_args])
    logger.info(f"args: {args}")
    main(args)
