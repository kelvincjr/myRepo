# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
from loguru import logger
import pandas as pd
import numpy as np
from collections import Counter,defaultdict

import sys
sys.path.append('../../../')

seg_len=0
seg_backoff=0
fold = 0

train_file = r'data/train.conll_convert.conll'
test_file = r'data/testA.conll_sent.conll'

ner_labels = ['器官组织', '属性', '阴性表现', '异常现象', '阳性表现', '修饰描述', '否定描述', '检查手段', '测量值', '数量', '疾病', '指代', '手术', '累及部位', '期象', '病理分级', '病理分型', '病理分期']

def load_ner_train_data(filename):
    D = []
    id_ = 0
    total_entities = 0
    total_spos = 0
    #predicate_counter = defaultdict(int)
    ner_counter = defaultdict(int)
    with open(filename, encoding='utf-8') as f:
        while True:
            l = f.readline()
            if not l.startswith('{'):
                break
            data = json.loads(l)
            sent = data['sent']
            ners = data['ners']
            guid = id_
            #print('id {}, sent {}'.format(id_, sent))
            entities = []
            for ner in ners:
                start_index = ner[0]
                end_index = ner[1]
                ner_type = ner[2]
                mention = ner[3]
                end_index = end_index - 1
                ner_counter[ner_type] += 1
                entities.append((ner_type, start_index, end_index))
                #entities = list(set(entities))
                D.append((guid, sent, entities))
                total_entities += 1
            id_ += 1
            l = f.readline()
            while l.startswith('[') or l.startswith('{'):
                spo = l.split('\t')
                sub = spo[0]
                obj = spo[1]
                pred = spo[2]
                #print('========== sub {}, pred {}, obj {}'.format(sub, pred, obj))
                l = f.readline()
                total_spos += 1
    
    print('total sent num: {}'.format(id_))
    sorted_ner = sorted(ner_counter.items(), key=lambda x: x[1], reverse=True)
    for s_ner, count in sorted_ner:
        print('{}, {}'.format(s_ner, count))  
    print('total entity num: {}, spo num: {}'.format(total_entities, total_spos))
    return D

def load_ner_test_data(filename):
    D = []
    id_ = 0
    with open(filename, encoding='utf-8') as f:
        while True:
            l = f.readline()
            if len(l) == 0:
                break
            sent = l.trim()
            guid = id_
            id_ += 1
            D.append((guid, sent, None))
    return D
        

#load_ner_train_data(train_file)
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
#import sys
#sys.exit()

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

    data = load_ner_train_data(train_file)

    for i, x in enumerate(tqdm(data)):
        guid = x[0]
        text = clean_text(x[1])
        sl = LabeledText(guid, text)
        entities = x[2]
        for entity in entities:
            c = entity[0]
            x0 = int(entity[1])
            x1 = int(entity[2])
            sl.add_entity(c, x0, x1)

        #print("index: ", str(i), ", text: ", text, ", entities: ", sl.entities)
        #if i > 10:
            #break
        yield guid, text, None, sl.entities

#train_data_generator(train_text_file, train_bio_file)
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

def load_eval_examples(eval_text_file, eval_bio_file):
    lines = []
    for guid, text, _, entities in train_data_generator(args.eval_file):
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

    data = load_ner_test_data(test_file, is_test=True)
    for i, x in enumerate(tqdm(data)):
        guid = x[0]
        text_a = clean_text(x[1])

        yield guid, text_a, None, None


def load_test_examples(args):
    test_base_examples = load_ner_examples(test_data_generator,
                                           args.test_file,
                                           seg_len=args.seg_len,
                                           seg_backoff=args.seg_backoff)

    logger.info(f"Loaded {len(test_base_examples)} test examples.")
    return test_base_examples

#train_data_generator(train_file)

def generate_submission(args):
    '''
    submission_file = f"ccks2021_predict.json"
    test_results = {}

    json.dump(test_results,
              open(f"{submission_file}", 'w'),
              ensure_ascii=False,
              indent=2)
    '''
    reviews_file = f"{args.latest_dir}/{args.dataset_name}_reviews_{args.local_id}.json"
    reviews = json.load(open(reviews_file, 'r'))

    submission_file = f"{args.submissions_dir}/{args.dataset_name}_submission_{args.local_id}.json"    
    D = []
    for guid, json_data in tqdm(reviews.items(), desc="events"):

        attributes = []
        #  output_data = {'doc_id': guid, 'events': []}
        text = json_data['text']
        id_ = json_data['guid']
        json_entities = json_data['entities']
        entities = []
        for json_entity in json_entities:
            category = json_entity['category']
            mention = json_entity['mention']
            start = int(json_entity['start'])
            end = int(json_entity['end'])
            entities.append((category, mention))
        D.append((text, id_, entities))

    rel_list = []
    with open(submission_file, 'w', encoding='utf-8') as f:
        for i in range(len(D)):
            texti, idxi, entities = D[i]
            for j in range(len(entities)):
                subject_type = entities[j][0]
                for k in range(len(entities)):
                    object_type = entities[k][0]
                    if j >= k or entities[j][1] == entities[k][1]:
                        continue
                    rel_list.append({"id":idxi, "text":texti, "spo_list":[{"subject": entities[j][1], "subject-type": subject_type, "object": entities[k][1], "object-type": object_type}]})
        json.dump(rel_list, f, ensure_ascii=False, indent=4)

    logger.info(f"Saved {len(reviews)} lines in {submission_file}")

    from theta.modeling import archive_local_model
    archive_local_model(args, submission_file)

from theta.modeling import Params, CommonParams, NerParams, NerAppParams, log_global_params

experiment_params = NerAppParams(
    CommonParams(
        dataset_name="xfyun",
        experiment_name="xfyun_ner",
        train_file=train_file,
        eval_file=None,
        test_file=test_file,#test_spo_file,
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
        num_augements=0,
        enable_kd=False,
        enable_sda=False,
        sda_teachers=2,
        loss_type="CrossEntropyLoss",
        #  loss_type='FocalLoss',
        focalloss_gamma=2.0,
        model_type="bert",
        model_path=
        #  "/opt/share/pretrained/pytorch/hfl/chinese-electra-large-discriminator",
        #r"/opt/kelvin/python/knowledge_graph/theta/theta-master-0826/theta/examples/LIC2021/model_rbt3",
        r"/kaggle/working",
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
        save_ner_preds(args, trainer.pred_results, test_examples)

    #do_predict(args)
    #print(trainer.pred_results)
    #generate_submission(args)

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