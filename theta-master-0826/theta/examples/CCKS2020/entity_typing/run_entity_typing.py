#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, random
from tqdm import tqdm
from loguru import logger
import pandas as pd
import numpy as np
import mlflow

from theta.modeling import load_glue_examples, show_glue_datainfo, save_glue_preds
from theta.modeling.glue import load_model, GlueTrainer, get_args

glue_labels = ['病毒', '细菌', '疾病', '药物', '医学专科', '检查科目', '症状', 'NoneType']
# 10,     52,    1267,  2550,  7,     147,   867,   100
# [0.200, 0.200, 0.025, 0.025, 0.200, 0.100, 0.050, 0.200]

# --- 0.9383 (10/10)
#[0.5, 0.40, 0.10, 0.05, 0.5, 0.40, 0.40, 0.40]

# -------------------- Data --------------------


def train_data_generator(train_file):
    df_train = pd.read_csv(train_file,
                           sep='\t',
                           header=None,
                           names=['text', 'label'])
    for i, row in tqdm(df_train.iterrows()):
        yield f"{i}", row.text, None, row.label


def load_train_val_examples(args):
    all_train_examples = load_glue_examples(train_data_generator,
                                            args.train_file)

    # 切分训练集和验证集
    # theta 提供split_train_eval_examples辅助函数
    from theta.utils import split_train_eval_examples
    train_examples, val_examples = split_train_eval_examples(
        all_train_examples,
        train_rate=args.train_rate,
        fold=args.fold,
        shuffle=True)

    #  random.shuffle(all_train_examples)
    #  num_train_examples = int(len(all_train_examples) * args.train_rate)
    #  val_examples = all_train_examples[num_train_examples:]
    #  train_examples = all_train_examples[:num_train_examples]

    logger.info(f"Loaded {len(train_examples)} train examples, "
                f"{len(val_examples)} val examples.")
    return train_examples, val_examples


def test_data_generator(test_file):
    for i, line in enumerate(tqdm(open(test_file, 'r'))):
        line = line.strip()
        yield f"{i}", line, None, None


def load_test_examples(args):
    test_base_examples = load_glue_examples(test_data_generator,
                                            args.test_file)
    logger.info(f"Loaded {len(test_base_examples)} test examples.")

    return test_base_examples


# -------------------- Trainer --------------------


class Trainer(GlueTrainer):
    def __init__(self, args, glue_labels):
        super(Trainer, self).__init__(args, glue_labels, build_model=None)

    #  def build_model(self, args):
    #      return build_model(args)

    def on_predict_end(self, args, test_dataset):
        super(Trainer, self).on_predict_end(args, test_dataset)
        logger.info(self.logits)
        logger.info(self.probs)
        logger.info(self.preds)


# -------------------- Output --------------------


def generate_submission(args):
    reviews_file = f"{args.latest_dir}/{args.dataset_name}_reviews_{args.local_id}.tsv"
    df_reviews = pd.read_csv(reviews_file, sep='\t')

    submission_file = f"{args.submissions_dir}/{args.dataset_name}_submission_{args.local_id}.txt"
    with open(submission_file, 'w') as fw:
        for i, row in tqdm(df_reviews.iterrows()):
            text = row.text_a
            label = row.label
            fw.write(f"{text}\t{label}\n")
    logger.info(f"Saved {df_reviews.shape[0]} lines in {submission_file}")

    #  ----- Tracking -----
    if args.do_experiment:
        mlflow.log_param("submission_file", submission_file)
        mlflow.log_artifact(submission_file)

    from theta.modeling import archive_local_model
    archive_local_model(args, submission_file)


#  def save_predict_results(args, pred_results, pred_results_file, test_examples):
#      assert len(test_examples) == len(pred_results)
#      with open(pred_results_file, 'w') as fw:
#          #  fw.write("id,y\n")
#          for input_example, v in zip(test_examples, pred_results):
#              #  ID = input_example.guid
#              text = input_example.text_a
#              v = args.id2label[v]
#              fw.write(f"{text}\t{v}\n")
#          logger.info(f"Result wrote to {pred_results_file}")
#
#      submission_file = pred_results_file
#      #  ----- Tracking -----
#      if args.do_experiment:
#          mlflow.log_param("submission_file", submission_file)
#          mlflow.log_artifact(submission_file)
#
#      from theta.modeling import archive_local_model
#      archive_local_model(args, submission_file)
#


def eda(args):
    show_glue_datainfo(glue_labels, train_data_generator, args.train_file,
                       test_data_generator, args.test_file)


from theta.modeling import Params, CommonParams, GlueParams, GlueAppParams, log_global_params

experiment_params = GlueAppParams(
    CommonParams(
        dataset_name="entity_typing",
        experiment_name="ccks2020_entity_typing",
        train_file="data/rawdata/ccks_7_1_competition_data/entity_type.txt",
        eval_file="data/rawdata/ccks_7_1_competition_data/entity_type.txt",
        test_file=
        "data/rawdata/ccks_7_1_competition_data/entity_validation.txt",
        learning_rate=1e-5,
        train_max_seq_length=32,
        eval_max_seq_length=32,
        per_gpu_train_batch_size=96,
        per_gpu_eval_batch_size=96,
        per_gpu_predict_batch_size=96,
        seg_len=0,
        seg_backoff=0,
        num_train_epochs=10,
        fold=8,
        num_augements=0,
        enable_kd=False,
        enable_sda=False,
        sda_teachers=3,
        sda_stategy="clone_models",
        sda_empty_first=True,
        #  loss_type="CrossEntropyLoss",
        loss_type="FocalLoss",
        focalloss_gamma=2.0,
        model_type="bert",
        model_path=
        "/opt/share/pretrained/pytorch/roberta-wwm-large-ext-chinese",
        train_rate=0.9,
        fp16=False,
        best_index='f1',
    ),
    GlueParams(glue_labels=glue_labels, ))

experiment_params.debug()

theta_commands = {}


def theta_command(fn):
    def wrapped(*args, **kwargs):
        logger.warning(f"{str(fn)}")
        logger.warning(f"args: {args}")
        logger.warning(f"kwargs: {kwargs}")
        return fn(*args, **kwargs)

    return wrapped


def do_submit(args):
    generate_submission(args)


@theta_command
def do_eda(args):
    show_glue_datainfo(glue_labels, train_data_generator, args.train_file,
                       test_data_generator, args.test_file)


def main(args):
    if args.do_eda:
        do_eda(args)

    elif args.do_submit:
        do_submit(args)

    else:

        class AppTrainer(GlueTrainer):
            def __init__(self, args, glue_labels):
                super(AppTrainer, self).__init__(args,
                                                 glue_labels,
                                                 build_model=None)

            #  def on_predict_end(self, args, test_dataset):
            #      super(Trainer, self).on_predict_end(args, test_dataset)

        trainer = AppTrainer(args, glue_labels)

        def do_train(args):
            train_examples, val_examples = load_train_val_examples(args)
            trainer.train(args, train_examples, val_examples)

        def do_eval(args):
            args.model_path = args.best_model_path
            _, eval_examples = load_train_val_examples(args)
            model = load_model(args)
            trainer.evaluate(args, model, eval_examples)

        def do_predict(args):
            args.model_path = args.best_model_path
            test_examples = load_test_examples(args)
            model = load_model(args)
            trainer.predict(args, model, test_examples)

            reviews_file = save_glue_preds(args, trainer.pred_results,
                                           test_examples)
            return reviews_file

        if args.do_train:
            do_train(args)

        elif args.do_eval:
            do_eval(args)

        elif args.do_predict:
            do_predict(args)

        elif args.do_experiment:
            if args.tracking_uri:
                mlflow.set_tracking_uri(args.tracking_uri)
            mlflow.set_experiment(args.experiment_name)

            with mlflow.start_run(run_name=f"{args.local_id}") as mlrun:
                log_global_params(args, experiment_params)

                # ----- Train -----
                do_train(args)

                # ----- Predict -----
                do_predict(args)

                # ----- Submit -----
                do_submit(args)


if __name__ == '__main__':

    def add_special_args(parser):
        return parser

    from theta.modeling.glue import load_model, get_args, GlueTrainer

    args = get_args(experiment_params=experiment_params,
                    special_args=[add_special_args])
    logger.info(f"args: {args}")

    #  args = experiment_params.update_args(args)

    main(args)
