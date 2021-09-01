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

train_labeled_spo_file = r'data/fixed_dev.json'
train2_labeled_spo_file = r'data/fixed_train.json'
test_spo_file = r'data/train/test.json'
train_unlabeled_spo_file = r'data/train/unlabeled.json'
schema_file = r'data/schema.csv'

ner_labels = ['机构', '灾害', '人物', '方案', '工作', '措施', '职责', '资源', '信息', '文件', '物资', '情况', '产品', '制度', '知识', '报告', '原因', '经验', '建议', '任务', '责任', '问题', '职务', '秩序', '区域', '设施', '能力', '捐赠', '支援', '预案', '交通', '意见', '行为', '电话', '会议', '标准', '计划', '响应', '行动', '命令', '决策', '公告', '支持', '机制', '疾病', '演练', '必需品', '意识', '资金', '指令', '政策', '部署', '体系', '事件', '影响', '状态', '物品', '事故', '建筑', '通知', '效果', '培训', '方法', '程序', '原则', '犯罪', '趋势', '分工']

def load_ner_train_data(filename, is_test=False):
    D = []
    num_lines = 0
    predicate_counter = defaultdict(int)
    ner_counter = defaultdict(int)
    with open(filename, encoding='utf-8') as f:
        #l = f.readlines()
        data_list = json.load(f)
        for data in tqdm(data_list):
            guid = data['id']
            text = clean_text(data['text'])
            #print('id: {}, text: {}, spo_list: {}'.format(guid, text, spo_list))
            entities = []
            if not is_test:
                spo_list = data['spo_list']
                for spo in spo_list:
                    sub_category = spo['subject-type']
                    sub_mention = spo['subject']
                    obj_category = spo['object-type']
                    obj_mention = spo['object']
                    sub_start = text.find(sub_mention)
                    obj_start = text.find(obj_mention)
                    predicate = spo['predicate']
                    predicate = sub_category+"_"+predicate+"_"+obj_category
                    predicate_counter[spo['predicate']] += 1
                    ner_counter[sub_category] += 1
                    ner_counter[obj_category] += 1
                    entities.append((sub_category, sub_start, sub_start+len(sub_mention) - 1))
                    entities.append((obj_category, obj_start, obj_start+len(obj_mention) - 1))      
            
            entities = list(set(entities))
            D.append((guid, text, entities))
            num_lines += 1
            
    return D

#load_texts(train_text_file)
#load_bioattr_labels(train_bio_file)
#ner_labels = read_schema(schema_file)
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

    data = load_ner_train_data(test_file, is_test=True)
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
        dataset_name="hualu",
        experiment_name="hualu_ner",
        train_file=train2_labeled_spo_file,
        eval_file=train_labeled_spo_file,
        test_file=train_labeled_spo_file,#test_spo_file,
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
        r"/opt/kelvin/python/knowledge_graph/theta/theta-master-0826/theta/examples/LIC2021/model_rbt3",
        #r"/kaggle/working",
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
        save_ner_preds(args, trainer.pred_results, test_examples)

    #do_predict(args)
    #print(trainer.pred_results)
    generate_submission(args)

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