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

train_text_file = r'E:\ai_contest\ccks\mytheta\myRepo\theta-master-0826\theta\examples\COMMODITY_NER\data\train\input.seq.char'
train_bio_file = r'E:\ai_contest\ccks\mytheta\myRepo\theta-master-0826\theta\examples\COMMODITY_NER\data\train\output.seq.bioattr'
schema_file = r'E:\ai_contest\ccks\mytheta\myRepo\theta-master-0826\theta\examples\COMMODITY_NER\data\vocab_attr.txt'
selectedKeys = ['品牌', '品类', '类型']

def load_texts(filename, cond):
    D = []
    num_lines = 0
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text = l.replace(' ', '').strip().replace("\n", '').replace(r'[SPA]',' ')
            if num_lines in cond:
                D.append(text)
            num_lines += 1
            if (num_lines % 1000) == 0:
                print('{} number of text lines read'.format(num_lines))
    return D

def load_bioattr_labels(filename, cond):
    D = []
    num_lines = 0
    key_bio_lines = {}
    with open(filename, encoding='utf-8') as f:
        for l in f:
            tags = l.split(' ')
            entity_start = False
            entity_start_index = -1
            entity_end_index = -1
            entity_type = ""
            cur_index = 0
            entities = []
            for tag in tags:
                if tag.startswith('B'):
                    if entity_start:
                        entity_end_index = cur_index - 1
                        if entity_type in cond:
                            entities.append((entity_type, entity_start_index, entity_end_index))
                        entity_start = False
                        entity_start_index = -1
                        entity_end_index = -1
                        entity_type = ""
                    entity_start = True
                    entity_type = tag[2:]
                    entity_start_index = cur_index
                elif tag.startswith('O'):
                    if entity_start:
                        entity_end_index = cur_index - 1
                        if entity_type in cond:
                            entities.append((entity_type, entity_start_index, entity_end_index))
                        entity_start = False
                        entity_start_index = -1
                        entity_end_index = -1
                        entity_type = ""
                cur_index += 1
            #print(D)
            #break
            if len(entities) > 0:
                D.append(entities)
                key_bio_lines[num_lines] = 1
            num_lines += 1
            if (num_lines % 1000) == 0:
                print('{} number of bio lines read'.format(num_lines))
    return key_bio_lines, D

def read_schema(filename):
    # 读取schema
    with open(filename, encoding='utf-8') as f:
        D = []
        for l in f:
            if l.strip() != 'null':
                D.append(l.strip())
        return D

#load_texts(train_text_file)
#load_bioattr_labels(train_bio_file)
ner_labels = read_schema(schema_file)
print(ner_labels)
'''
train_data = load_data(train_file)
valid_data = load_data(dev_file)
id2label, label2id, num_labels = read_baiduee_schema(schema_file)

ner_labels = [
    f"{key[0]}_{key[1]}" for n, key in id2label.items()
]
'''
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


def train_data_generator(train_text_file, train_bio_file):

    texts = load_texts(train_text_file)
    cond, labels = load_bioattr_labels(train_bio_file)

    for i, x in enumerate(tqdm(texts)):
        guid = str(i)
        text = clean_text(x)
        sl = LabeledText(guid, text)
        entities = labels[i]
        for entity in entities:
            c = entity[0]
            x0 = int(entity[1])
            x1 = int(entity[2])
            sl.add_entity(c, x0, x1)

        #print("index: ", str(i), ", text: ", text, ", entities: ", sl.entities)
        #if i > 10:
            #break
        yield str(i), text, None, sl.entities

#train_data_generator(train_text_file, train_bio_file)
#sys.exit(0)

def load_train_val_examples(args):
    lines = []
    for guid, text, _, entities in train_data_generator(args.train_file, args.eval_file):
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

def load_eval_examples(eval_text_file, eval_bio_file):
    lines = []
    for guid, text, _, entities in train_data_generator(eval_text_file, eval_bio_file):
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
        num_augements=0,
        allow_overlap=allow_overlap)

    eval_examples = train_base_examples

    logger.info(f"Loaded {len(eval_examples)} eval examples")
    return eval_examples

def test_data_generator(test_file):

    texts = load_texts(train_text_file)
    for i, x in enumerate(tqdm(texts)):
        guid = str(i)
        text_a = clean_text(x)

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
        experiment_name="commodity_ner",
        train_file=train_text_file,
        eval_file=train_bio_file,
        #test_file=test_file,
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
    #NerParams(ner_labels=ner_labels, ner_type='span'))
    NerParams(ner_labels=ner_labels, ner_type='crf', no_crf_loss=False))

experiment_params.debug()

# -------------------- Trainer --------------------
class AppTrainer(NerTrainer):
    def __init__(self, args, ner_labels):
        super(AppTrainer, self).__init__(args, ner_labels, build_model=None)


def main(args):

    trainer = AppTrainer(args, ner_labels)
    #train_examples, val_examples = load_train_val_examples(args)
    #trainer.train(args, train_examples, val_examples)
    def do_eval(args):
        args.model_path = args.best_model_path
        eval_examples = load_eval_examples(args.eval_file)
        model = load_model(args)
        trainer.evaluate(args, model, eval_examples)

    def do_predict(args):
        args.model_path = args.best_model_path
        test_examples = load_test_examples(args)
        model = load_model(args)
        trainer.predict(args, model, test_examples)

    do_predict(args)
    print(trainer.pred_results)

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