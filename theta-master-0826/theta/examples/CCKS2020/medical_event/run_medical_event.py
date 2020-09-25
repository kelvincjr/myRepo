#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, random, copy, re
from tqdm import tqdm
from loguru import logger
from pathlib import Path
import pandas as pd
import mlflow

from theta.utils import load_json_file, split_train_eval_examples
from theta.modeling import LabeledText, load_ner_examples, load_ner_labeled_examples, save_ner_preds, show_ner_datainfo

#  if os.environ['NER_TYPE'] == 'span':
#      from theta.modeling.ner_span import load_model, get_args
#  else:
#      from theta.modeling.ner import load_model, get_args

ner_labels = ['肿瘤部位', '病灶大小', '转移部位']


# -------------------- Data --------------------
def clean_text(text):
    if text:
        text = text.strip()
        #  text = re.sub('\t', ' ', text)
        # 将7.32*8.41CM 转换成7.32CMx8.41CM，官方数据格式隐含要求。
        text = re.sub('([\d\.]+)\s*\*\s*([\d\.]+)\s*CM', r"\1CM×\2CM", text)
        text = re.sub('([\d\.]+)CM\*([\d\.]+)CM', r"\1CM×\2CM", text)
        text = re.sub('([\d\.]+)×([\d\.]+)CM', r"\1CM×\2CM", text)

        text = re.sub('([\d\.]+)\s*\*\s*([\d\.]+)\s*cm', r"\1CM×\2CM", text)
        text = re.sub('([\d\.]+)cm\*([\d\.]+)cm', r"\1CM×\2CM", text)
        text = re.sub('([\d\.]+)×([\d\.]+)cm', r"\1CM×\2CM", text)
        text = re.sub('([\d\.]+)cm×([\d\.]+)cm', r"\1CM×\2CM", text)

        text = re.sub('([\d\.]+)\s*\*\s*([\d\.]+)\s*mm', r"\1MM×\2MM", text)
        text = re.sub('([\d\.]+)cm\*([\d\.]+)mm', r"\1MM×\2MM", text)
        text = re.sub('([\d\.]+)×([\d\.]+)mm', r"\1MM×\2MM", text)
        text = re.sub('([\d\.]+)mm×([\d\.]+)mm', r"\1MM×\2MM", text)
    return text


def read_train_file(train_file):
    lines = []

    df_train = pd.read_csv(train_file, sep='\t')
    df_train = df_train.fillna("")

    for i, row in tqdm(df_train.iterrows(), desc="Load train"):
        text = clean_text(row.原文)
        #  logger.warning(f"text: {text}")

        primary_site = row.肿瘤原发部位.strip()
        focus_size = row.原发病灶大小.strip()
        metastatic_site = row.转移部位.strip()

        #  logger.debug(
        #      f"{i} - primary_site: {primary_site}, focus_size: {focus_size}, metastatic_site: {metastatic_site}"
        #  )
        entities = []
        if primary_site:
            sites = primary_site.split(',')
            for ps in sites:
                ps = ps.strip()
                if not ps:
                    continue

                p0 = text.find(ps)
                #  logger.info(f"primary_site: {primary_site}")
                #  logger.info(f"p0: {p0}")
                if p0 >= 0:
                    s = p0
                    e = p0 + len(ps)
                    entities.append({
                        "label_type": "肿瘤部位",
                        "start_pos": s,
                        "end_pos": e,
                        "mention": text[s:e]
                    })
                #  else:
                #      logger.warning(
                #          f"[{i}] - primary_size: {ps} not found in text: {text}"
                #      )

        if focus_size:
            sizes = focus_size.split(',')
            for fs in sizes:
                fs = fs.strip()
                if not fs:
                    continue
                p0 = text.find(fs)
                if p0 >= 0:
                    s = p0
                    e = p0 + len(fs)
                    entities.append({
                        "label_type": "病灶大小",
                        "start_pos": s,
                        "end_pos": e,
                        "mention": text[s:e]
                    })
                #  else:
                #      logger.warning(
                #          f"[{i}] - focus_size: {fs} not found in text: {text}")

        if metastatic_site:
            sites = metastatic_site.split(',')
            s0 = 0
            for ms in sites:
                ms = ms.strip()
                if not ms:
                    continue
                if ms == primary_site:
                    #  logger.warning(
                    #      f"[{i}] - Skip! metastatic_site: {ms} == primary_size: {primary_site} in text: {text}"
                    #  )
                    continue
                if ms in primary_site:
                    #  logger.warning(
                    #      f"[{i}] - Skip! metastatic_site: {ms} contained in  primary_size: {primary_site} in text: {text}"
                    #  )
                    continue
                if primary_site in ms:
                    #  logger.warning(
                    #      f"[{i}] - Skip! primary_size: {primary_site} contained in metastatic_site: {ms} in text: {text}"
                    #  )
                    continue

                p0 = text.find(ms, s0)
                if p0 >= 0:
                    s = p0
                    e = p0 + len(ms)
                    #  if s == 128 and e == 129:
                    #      logger.debug(f"{i} : (){len(text)}) {text}")
                    #      logger.warning(
                    #          f"metastatic_site: {metastatic_site}, s: {s}, e: {e}, mention: {text[s:e]}"
                    #      )
                    entities.append({
                        "label_type": "转移部位",
                        "start_pos": s,
                        "end_pos": e,
                        "mention": text[s:e]
                    })
                    s0 = p0 + len(ms)
                #  else:
                #      logger.warning(
                #          f"[{i}] - metastatic_site: {ms} not found in text: {text}"
                #      )

        #  logger.debug(f"entities: {entities}")
        if entities:
            lines.append({"originalText": text, "entities": entities})

    #  aug_tokens = [(-1, x['originalText'], x['entities']) for x in lines]
    #  for guid, text, entities in aug_tokens:
    #      text_a = text
    #      for entity in entities:
    #          logger.debug(f"text_a: {text_a}")
    #          logger.debug(
    #              f"text_a[entity['start_pos']:entity['end_pos']]: {text_a[entity['start_pos']:entity['end_pos']]}"
    #          )
    #          logger.debug(
    #              f"mention {entity['mention']} in {text_a.find(entity['mention'])}"
    #          )
    #          logger.debug(f"entity: {entity}")
    #          assert text_a[entity['start_pos']:entity['end_pos']] == entity[
    #              'mention']

    return lines


def train_data_generator(train_file):

    lines = read_train_file(train_file)

    for i, x in enumerate(tqdm(lines)):
        guid = str(i)
        text = clean_text(x['originalText'])
        sl = LabeledText(guid, text)

        entities = x['entities']
        for entity in entities:
            start_pos = entity['start_pos']
            end_pos = entity['end_pos'] - 1
            category = entity['label_type']
            sl.add_entity(category, start_pos, end_pos)

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

    df_test = pd.read_csv(test_file, sep='\t')
    df_test = df_test.fillna("")

    for i, row in tqdm(df_test.iterrows(), desc="Load test"):
        text_a = clean_text(row.原文)
        guid = str(i)

        yield guid, text_a, None, None


def load_test_examples(args):
    test_base_examples = load_ner_examples(test_data_generator,
                                           args.test_file,
                                           seg_len=args.seg_len,
                                           seg_backoff=args.seg_backoff)

    logger.info(f"Loaded {len(test_base_examples)} test examples.")
    return test_base_examples


def generate_submission(args):
    #  submission_file = os.path.join(args.output_dir,
    #                                 f"{args.dataset_name}_submission.xlsx")
    submission_file = f"{args.submissions_dir}/{args.dataset_name}_submission_{args.local_id}.xlsx"
    import xlwt
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet(f"{args.dataset_name}")

    worksheet.write(0, 0, label='原文')
    worksheet.write(0, 1, label='肿瘤原发部位')
    worksheet.write(0, 2, label='原发病灶大小')
    worksheet.write(0, 3, label='转移部位')

    reviews_file = f"{args.latest_dir}/{args.dataset_name}_reviews_{args.local_id}.json"
    reviews = json.load(open(reviews_file, 'r'))

    idx = 1
    for guid, json_data in reviews.items():
        text = json_data['text']
        entities = json_data['entities']
        label_entities = {}
        for entity in entities:
            c = entity['category']
            s = entity['start']
            e = entity['end'] + 1
            entity_text = text[s:e]

            if s > len(text) or e > len(text):
                continue
            if len(entity_text) == 0 or len(entity_text) > 16:
                continue
            if ';' in entity_text or '、' in entity_text:
                continue

            if c not in label_entities:
                label_entities[c] = []
            label_entities[c].append(entity_text)

        worksheet.write(idx, 0, label=text)
        if '肿瘤部位' in label_entities:
            worksheet.write(idx, 1, ','.join(label_entities['肿瘤部位']))
        if '病灶大小' in label_entities:
            worksheet.write(idx, 2, ','.join(label_entities['病灶大小']))
        if '转移部位' in label_entities:
            worksheet.write(idx, 3, ','.join(label_entities['转移部位']))

        idx += 1

    workbook.save(submission_file)

    if args.do_experiment:
        mlflow.log_param("submission_file", submission_file)
        mlflow.log_artifact(submission_file)

    logger.info(f"Saved {submission_file}")

    from theta.modeling import archive_local_model
    archive_local_model(args, submission_file)


from theta.modeling import Params, CommonParams, NerParams, NerAppParams, log_global_params

experiment_params = NerAppParams(
    CommonParams(
        dataset_name="medical_event",
        experiment_name="ccks2020_medical_event",
        tracking_uri="http://tracking.mlflow:5000",
        train_file='data/task2_train_reformat.tsv',
        eval_file='data/task2_train_reformat.tsv',
        test_file='data/task2_no_val.tsv',
        learning_rate=2e-5,
        train_max_seq_length=512,
        eval_max_seq_length=512,
        per_gpu_train_batch_size=4,
        per_gpu_eval_batch_size=4,
        per_gpu_predict_batch_size=4,
        seg_len=510,
        seg_backoff=128,
        num_train_epochs=10,
        fold=3,
        num_augements=2,
        enable_kd=True,
        loss_type="CrossEntropyLoss",
        model_type="bert",
        model_path=
        "/opt/share/pretrained/pytorch/roberta-wwm-large-ext-chinese",
        fp16=True,
        random_type='np',
    ), NerParams(ner_labels=ner_labels, ner_type='crf', no_crf_loss=True))

experiment_params.debug()


def main(args):
    def do_eda(args):
        show_ner_datainfo(ner_labels, train_data_generator, args.train_file,
                          test_data_generator, args.test_file)

    def do_submit(args):
        generate_submission(args)

    if args.do_eda:
        do_eda(args)

    elif args.do_submit:
        do_submit(args)

    elif args.to_train_poplar:
        from theta.modeling import to_train_poplar
        to_train_poplar(args,
                        train_data_generator,
                        ner_labels=ner_labels,
                        ner_connections=[],
                        start_page=args.start_page,
                        max_pages=args.max_pages)

    elif args.to_reviews_poplar:
        from theta.modeling import to_reviews_poplar
        to_reviews_poplar(args,
                          ner_labels=ner_labels,
                          ner_connections=[],
                          start_page=args.start_page,
                          max_pages=args.max_pages)
    else:
        # -------------------- Model --------------------
        if args.ner_type == 'span':
            from theta.modeling.ner_span import NerTrainer
        else:
            from theta.modeling.ner import NerTrainer

        class AppTrainer(NerTrainer):
            def __init__(self, args, ner_labels):
                super(AppTrainer, self).__init__(args,
                                                 ner_labels,
                                                 build_model=None)

            #  def on_predict_end(self, args, test_dataset):
            #      super(Trainer, self).on_predict_end(args, test_dataset)

        trainer = AppTrainer(args, ner_labels)

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
            reviews_file, category_mentions_file = save_ner_preds(
                args, trainer.pred_results, test_examples)
            return reviews_file, category_mentions_file

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
        parser.add_argument("--to_train_poplar", action="store_true")
        parser.add_argument("--to_reviews_poplar", action="store_true")
        parser.add_argument("--start_page", type=int, default=0)
        parser.add_argument("--max_pages", type=int, default=100)
        return parser

    if experiment_params.ner_params.ner_type == 'span':
        from theta.modeling.ner_span import load_model, get_args, NerTrainer
    else:
        from theta.modeling.ner import load_model, get_args, NerTrainer

    args = get_args(experiment_params=experiment_params,
                    special_args=[add_special_args])
    logger.info(f"args: {args}")

    main(args)
