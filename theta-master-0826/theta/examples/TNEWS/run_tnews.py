#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
from loguru import logger
import numpy as np

from theta.modeling import load_glue_examples
from theta.modeling.glue import GlueTrainer, load_model, get_args
from theta.utils import load_json_file

# -------------------- Data --------------------

# ## 1. 数据观察
train_file = './data/rawdata/train.json'
test_file = './data/rawdata/test.json'
eval_file = './data/rawdata/dev.json'
labels_file = './data/rawdata/labels.json'

# ### 1.1 样本数量分布
train_data = load_json_file(train_file)
test_data = load_json_file(test_file)
eval_data = load_json_file(eval_file)

all_data = train_data + eval_data
descs = [x['label_desc'] for x in all_data]
from collections import Counter
logger.debug(f"{Counter(descs)}")

# ### 1.2 样本长度分布
lengths = [len(x['sentence']) for x in all_data]
logger.info(f"***** Text Lengths *****")
logger.info(f"mean: {np.mean(lengths):.2f}")
logger.info(f"std: {np.mean(lengths):.2f}")
logger.info(f"max: {np.max(lengths)}")
logger.info(f"min: {np.min(lengths)}")

# ### 1.3 样本标签
labels_data = load_json_file(labels_file)
label2desc = {x['label']: x['label_desc'] for x in labels_data}
desc2label = {x['label_desc']: x['label'] for x in labels_data}

glue_labels = [x['label_desc'] for x in labels_data]
logger.info(f"glue_labels: {len(glue_labels)} {glue_labels}")

# -------------------- Model --------------------

# ## 2. 模型构建

# ### 2.1 模型输入数据


def clean_text(text):
    text = text.strip().replace('\n', '')
    text = text.replace('\t', ' ')
    return text


def train_data_generator(train_file):
    train_data = load_json_file(train_file)
    for i, json_data in enumerate(tqdm(train_data, desc="train")):
        guid = str(i)
        text = json_data['sentence']
        text = clean_text(text)
        label = json_data['label_desc']

        yield guid, text, None, label


def eval_data_generator(eval_file):
    eval_data = load_json_file(eval_file)
    for i, json_data in enumerate(tqdm(eval_data, desc="eval")):
        guid = str(i)
        text = json_data['sentence']
        text = clean_text(text)
        label = json_data['label_desc']

        yield guid, text, None, label


def test_data_generator(test_file):
    test_data = load_json_file(test_file)
    total_examples = len(test_data)
    for i, json_data in enumerate(tqdm(test_data, desc="test")):
        guid = str(json_data['id'])
        text = json_data['sentence']
        text = clean_text(text)

        yield guid, text, None, None


# ### 2.2 模型输出结果
def save_predict_results(args, pred_results, pred_results_file, test_examples):
    descs = []
    with open(pred_results_file, 'w') as wr:
        for label, example in zip(pred_results, test_examples):
            label_desc = args.id2label[label]
            descs.append(label_desc)
            ID = example.guid
            text = example.text_a
            json_data = {
                'id': ID,
                'label': desc2label[label_desc],
                'label_desc': label_desc,
                'sentence': text
            }
            wr.write(f"{json.dumps(json_data, ensure_ascii=False)}\n")
    logger.info(f"Predict results file saved to {pred_results_file}")
    from collections import Counter
    print(f"{Counter(descs)}")


# ### 2.3 数据样本集合
def load_train_examples(train_file):
    train_examples = load_glue_examples(train_data_generator, train_file)
    logger.info(f"Loaded {len(train_examples)} train examples.")

    return train_examples


def load_eval_examples(eval_file):
    eval_examples = load_glue_examples(eval_data_generator, eval_file)
    logger.info(f"Loaded {len(eval_examples)} eval examples.")

    return eval_examples


def load_test_examples(test_file):
    test_examples = load_glue_examples(test_data_generator, test_file)
    logger.info(f"Loaded {len(test_examples)} test examples.")

    return test_examples


# 自定义训练器


class AppTrainer(GlueTrainer):
    def __init__(self, args, glue_labels):
        # 使用自定义模型时，传入build_model参数。
        super(AppTrainer, self).__init__(args, glue_labels, build_model=None)


# 主控流程


def main(args):

    if args.do_eda:
        from theta.modeling import show_glue_datainfo
        show_glue_datainfo(glue_labels, train_data_generator, args.train_file,
                           test_data_generator, args.test_file)
    else:
        trainer = AppTrainer(args, glue_labels)

        # --------------- Train ---------------
        if args.do_train:
            train_examples = load_train_examples(args.train_file)
            eval_examples = load_eval_examples(args.eval_file)
            trainer.train(args, train_examples, eval_examples)

        # --------------- Evaluate ---------------
        elif args.do_eval:
            eval_examples = load_eval_examples(args.eval_file)
            model = load_model(args)
            trainer.evaluate(args, model, eval_examples)

        # --------------- Predict ---------------
        elif args.do_predict:
            test_examples = load_test_examples(args)
            model = load_model(args)
            trainer.predict(args, model, test_examples)

            save_predict_results(args, trainer.pred_results,
                                 f"./{args.dataset_name}_predict.json",
                                 test_examples)


if __name__ == '__main__':

    def add_special_args(parser):
        parser.add_argument("--do_eda", action="store_true")
        return parser

    from theta.modeling.glue import get_args
    args = get_args([add_special_args])
    main(args)
