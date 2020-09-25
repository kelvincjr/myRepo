#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, random, copy
from tqdm import tqdm
from loguru import logger
from pathlib import Path

from theta.utils import load_json_file, split_train_eval_examples
from theta.modeling import LabeledText, load_ner_examples, load_ner_labeled_examples, save_ner_preds, show_ner_datainfo

if os.environ['NER_TYPE'] == 'span':
    from theta.modeling.ner_span import load_model, NerTrainer, get_args
else:
    from theta.modeling.ner import load_model, NerTrainer, get_args

#  地址（address）
#  书名（book）
#  公司（company）
#  游戏（game）
#  政府（goverment）
#  电影（movie）
#  姓名（name）
#  组织机构（organization）
#  职位（position）
#  景点（scene）

ner_labels = [
    'address', 'book', 'company', 'game', 'government', 'movie', 'name',
    'organization', 'position', 'scene'
]
logger.info(f"ner_labels: {ner_labels}")


# -------------------- Data --------------------
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

        # -------------------- 训练数据json格式 --------------------
        #  {
        #      "text": "万通地产设计总监刘克峰；",
        #      "label": {
        #          "name": {
        #              "刘克峰": [[8, 10]]
        #          },
        #          "company": {
        #              "万通地产": [[0, 3]]
        #          },
        #          "position": {
        #              "设计总监": [[4, 7]]
        #          }
        #      }
        #  }

        entities = []
        classes = x['label'].keys()
        for c in classes:
            c_labels = x['label'][c]
            #  logger.debug(f"c_labels:{c_labels}")
            for label, span in c_labels.items():
                x0, x1 = span[0]
                sl.add_entity(c, x0, x1)

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
        shuffle=True,
        random_state=args.seed)

    logger.info(f"Loaded {len(train_examples)} train examples, "
                f"{len(val_examples)} val examples.")
    return train_examples, val_examples


def test_data_generator(test_file):

    lines = load_json_file(test_file)
    for i, s in enumerate(tqdm(lines)):
        guid = str(i)
        text_a = clean_text(s['text'])

        yield guid, text_a, None, None


def load_test_examples(args):
    test_base_examples = load_ner_examples(test_data_generator,
                                           args.test_file,
                                           seg_len=args.seg_len,
                                           seg_backoff=args.seg_backoff)

    logger.info(f"Loaded {len(test_base_examples)} test examples.")
    return test_base_examples


def generate_submission(args):
    reviews_file = f"{args.output_dir}/{args.dataset_name}_reviews_fold{args.fold}.json"
    reviews = json.load(open(reviews_file, 'r'))

    submission_file = f"{args.dataset_name}_predict.json"
    test_results = {}
    for guid, json_data in reviews.items():
        text = json_data['text']

        if guid not in test_results:
            test_results[guid] = {
                "guid": guid,
                "content": "",
                "events": [],
                "tagged_text": "",
            }

        s0 = 0
        tagged_text = test_results[guid]['tagged_text']
        for json_entity in json_data['entities']:
            event_type = json_entity['category']
            entity_text = json_entity['mention']
            s = json_entity['start']
            e = json_entity['end']
            test_results[guid]['events'].append(
                (event_type, entity_text, s, e))

            tagged_text += f"{text[s0:s]}\n"
            tagged_text += f"【{event_type} | {entity_text}】\n"
            test_results[guid]['tagged_text'] = tagged_text
            test_results[guid]['content'] += text

            s0 = e

        test_results[guid]['events'] = sorted(test_results[guid]['events'],
                                              key=lambda x: x[3])

    json.dump(test_results,
              open(f"{submission_file}", 'w'),
              ensure_ascii=False,
              indent=2)

    logger.info(f"Saved {len(reviews)} lines in {submission_file}")


# -------------------- Trainer --------------------
class AppTrainer(NerTrainer):
    def __init__(self, args, ner_labels):
        super(AppTrainer, self).__init__(args, ner_labels, build_model=None)


def main(args):

    if args.generate_submission:
        generate_submission(args)
    else:
        trainer = AppTrainer(args, ner_labels)

        if args.do_eda:
            show_ner_datainfo(ner_labels, train_data_generator,
                              args.train_file, test_data_generator,
                              args.test_file)

        elif args.do_train:
            train_examples, val_examples = load_train_val_examples(args)
            trainer.train(args, train_examples, val_examples)

        elif args.do_eval:
            _, eval_examples = load_train_val_examples(args)
            model = load_model(args)
            trainer.evaluate(args, model, eval_examples)

        elif args.do_predict:
            test_examples = load_test_examples(args)
            model = load_model(args)
            trainer.predict(args, model, test_examples)
            save_ner_preds(args, trainer.pred_results, test_examples)


if __name__ == '__main__':

    def add_special_args(parser):
        parser.add_argument("--do_eda", action="store_true", help="")
        parser.add_argument("--generate_submission",
                            action="store_true",
                            help="")
        return parser

    args = get_args([add_special_args])
    main(args)
