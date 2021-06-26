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

train_file = r'/kaggle/working/thetamaster3/theta-master-0826/theta/examples/CCKS2021/data/ccks_task1_train.txt'
eval_file = r'/kaggle/working/thetamaster3/theta-master-0826/theta/examples/CCKS2021/data/ccks_task1_eval_data.txt'

def load_data(filename):
    D = []
    type_set = set()
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            level1 = l['level1']
            level2 = l['level2']
            level3 = l['level3']
            attributes = []
            if 'attributes' in l:
                for attr in l['attributes']:
                        start = int(attr['start'])
                        end = int(attr['end'])
                        entity = attr['entity']
                        en_type = attr['type']
                        temp = level1 + "_" + level2 + "_" + level3 + "_" + en_type
                        type_set.add(temp)
                        value = (start, end, en_type, entity)
                        attributes.append(value)
            D.append((l['text'], l['text_id'], level1, level2, level3, attributes))
    return D, type_set

D, type_set = load_data(train_file)
ner_labels = list(type_set)
print(ner_labels)
print("labels len: ", len(ner_labels))
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

#from theta.modeling.ner_span import load_model, NerTrainer, get_args
from theta.modeling.ner import load_model, NerTrainer, get_args

def clean_text(text):
    if text:
        text = text.strip()
        #  text = re.sub('\t', ' ', text)
    return text


def train_data_generator(train_file):

    data, _ = load_data(train_file)

    for i, x in enumerate(tqdm(data)):
        guid = x[1]
        text = clean_text(x[0])
        level1 = x[2]
        level2 = x[3]
        level3 = x[4]
        sl = LabeledText(guid, text)
        entities = x[5]
        for entity in entities:
            c = level1 + "_" + level2 + "_" + level3 + "_" + entity[2]
            x0 = int(entity[0])
            x1 = int(entity[1])
            sl.add_entity(c, x0, x1)

        #print("index: ", str(i), ", text: ", text, ", entities: ", sl.entities)
        #if i > 10:
            #break
        yield str(i), text, None, sl.entities

#train_data_generator(train_file)
#sys.exit(0)

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

def load_eval_examples(eval_file):
    lines = []
    for guid, text, _, entities in train_data_generator(eval_file):
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

    data, _ = load_data(test_file)
    for i, x in enumerate(tqdm(data)):
        guid = x[1]
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
        dataset_name="ccks2021",
        experiment_name="ner",
        train_file=train_file,
        eval_file=eval_file,
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
        num_train_epochs=5,
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
        # r"E:\ai_contest\ccks\mytheta\myRepo\theta-master-0826\theta\examples\LIC2021\model_rbt3",
        "/kaggle/working",
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
    train_examples, val_examples = load_train_val_examples(args)
    trainer.train(args, train_examples, val_examples)
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

    #do_predict(args)
    #print(trainer.pred_results)

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