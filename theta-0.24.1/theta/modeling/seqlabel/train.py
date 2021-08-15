# -*- coding: utf-8 -*-
#%% ------------------------------------------------------------
import os, sys
import re, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#  import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline
#  sns.set(style='white', context='notebook', palette='deep')

import tensorflow as tf
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.callbacks import Callback, EarlyStopping

from loguru import logger
from tqdm import tqdm
from pprint import pprint
from multiprocessing import Process

from train_config import Config_WORKCONTENT, Config_PROJRESP
from train_config import get_best_model_weights_file_name, get_wrong_eval_file_name
from seqlabel_utils import ensure_empty_directory, list_find
from tf_utils import get_gpu_info
from data import load_train_data
from data import load_cross_validation_config, save_cross_validation_config
from data import get_cross_validation_config, get_train_dev_data
from seqlabel_model import BertTokenizer, BertModel

from seqlabel_utils import longest_common_substring

if len(sys.argv) > 1:
    c = sys.argv[1]
else:
    c = ""
if c not in ['WORKCONTENT', 'PROJRESP']:
    print(
        f"Usage: python prepare_seqlabel_dataset.py <'WORKCONTENT'|'PROJRESP'>"
    )
    exit - 1

if c == "WORKCONTENT":
    Config = Config_WORKCONTENT
else:
    Config = Config_PROJRESP
Config.category = c

raw_train_data_file = f"{Config.data_dir}/train_{Config.category}{Config.surfix}.tsv"
raw_test_data_file = f"{Config.data_dir}/test_{Config.category}{Config.surfix}.tsv"
train_folds_file = f"{Config.output_dir}/train_{Config.category}_{Config.total_folds}folds{Config.surfix}.pkl"

np.random.seed(Config.SEED)

logger.add(f"log/train_seqlabel_{Config.experiment}_{Config.category}.log")


class TrainData():
    def __init__(self, data, tokenizer, max_seq_length, batch_size=32):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.tag_tokens = self._tokenize(self.data, self.tokenizer,
                                         max_seq_length)
        self.max_seq_length = max_seq_length

    def _tokenize(self, data, tokenizer, max_seq_length):
        tag_tokens = []
        for i, d in enumerate(data):
            #  text, c = d[0][:max_seq_length], d[1]
            text, c = d[0], d[1]
            #  text += " " * (max_seq_length - len(text))
            text = u'___%s___%s' % (c, text)
            text = text[:max_seq_length]

            tokens = tokenizer.tokenize(text)
            num_tokens = len(tokens)

            e = d[2]
            e_tokens = tokenizer.tokenize(e)[1:-1]
            s1, s2 = np.zeros(num_tokens), np.zeros(num_tokens)
            start = list_find(tokens, e_tokens)
            if start != -1:
                end = start + len(e_tokens) - 1
                #  print(f"start:{start}, end:{end}")
                s1[start] = 1
                s2[end] = 1
            tag_tokens.append((text, s1.tolist(), s2.tolist()))
        return tag_tokens

    def __len__(self):
        return self.steps

    def __iter__(self, max_seq_length):
        def seq_padding(X, padding=0):
            L = [len(x) for x in X]
            ML = max(L)
            return np.array([
                np.concatenate([x, [padding] *
                                (ML - len(x))]) if len(x) < ML else x
                for x in X
            ])

        while True:
            idxs = [x for x in range(len(self.data))]
            np.random.shuffle(idxs)
            X1, X2, S1, S2 = [], [], [], []
            for i in idxs:
                text, s1, s2 = self.tag_tokens[i]
                if not (s1 is None or s2 is None):
                    x1, x2 = self.tokenizer.encode(first=text)
                    X1.append(x1)
                    X2.append(x2)
                    S1.append(np.array(s1))
                    S2.append(np.array(s2))
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        #  print(f"X1, X2, S1, S2")
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        yield [X1, X2, S1, S2], None
                        X1, X2, S1, S2 = [], [], [], []


#%% ------------------------------------------------------------
# Evoluation
class Evaluate(Callback):
    def __init__(self, bert_dev_data, extract_entity, bert_model,
                 learning_rate, min_learning_rate, strictly_equal):
        self.bert_dev_data = bert_dev_data
        self.extract_entity = extract_entity
        self.bert_model = bert_model
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.strictly_equal = strictly_equal
        #  self.train_ACC = []
        self.dev_ACC = []
        #  self.train_best = 0.
        self.best_loss = float('inf')
        self.last_loss = float('inf')
        self.best_f1 = 0.0
        self.last_f1 = 0.0
        self.num_loss_up = 0
        self.passed = 0

    def on_batch_begin(self, batch, logs=None):

        if 'COLAB_TPU_ADDR' in os.environ:
            self.on_epoch_begin_tpu(batch, logs)
            return
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * self.learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (
                self.learning_rate - self.min_learning_rate)
            lr += self.min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        #  print(
        #      f"on_batch_begin() learning rate: {lr:.6f}, passed: {self.passed}, params['steps']: {self.params['steps']}"
        #  )

    def get_dev_loss(self, predict_tokens):
        # ------- loss -------
        def categorical_crossentropy_loss(y_true: np.array,
                                          y_pred: np.array) -> float:
            """
            计算分类交叉熵（适用于多分类，张量版本）
            y_true: 训练标签，[[1,0,0], [0,1,0],...]。
            y_pred: 预测标签，[[1,0,0], [0,1,0],...]。
            返回交叉熵张量，形如[0.983423]。
            """

            #  # 传入单个标签时，转换成标签列表的形式。
            #  if len(y_true.shape) == 1:
            #      y_true = y_true.reshape((1, y_true.shape[0]))
            #      y_pred = y_pred.reshape((1, y_pred.shape[0]))
            # 生成形如[[0,1,0,...], [0,0,1,...],...]的张量。
            y_true = tf.convert_to_tensor(y_true, dtype='float32')
            y_pred = tf.convert_to_tensor(y_pred, dtype='float32')

            # 根据标签位置距离，计算分类权重。
            i_true = K.argmax(y_true, axis=1)
            i_pred = K.argmax(y_pred, axis=1)
            distance = K.abs(i_true - i_pred)
            length = K.int_shape(y_pred)[1] - 1
            weights = K.cast(distance / length, dtype='float32')

            losses = ((1.0 + weights) *
                      categorical_crossentropy(y_true, y_pred))
            loss = K.eval(K.mean(losses))
            return loss

        subject_tokens = self.bert_dev_data.tag_tokens

        max_s1 = max(len(x) for _, x, _ in subject_tokens)
        max_s2 = max(len(x) for _, _, x in subject_tokens)
        max_k1 = max(len(x) for x, _ in predict_tokens)
        max_k2 = max(len(x) for _, x in predict_tokens)
        max_tokens_len = max([max_s1, max_s2, max_k1, max_k2])

        s1 = []
        s2 = []
        for _, _s1, _s2 in subject_tokens:
            s1.append(_s1 + [0.0] * (max_tokens_len - len(_s1)))
            s2.append(_s2 + [0.0] * (max_tokens_len - len(_s2)))

        k1 = []
        k2 = []
        for _k1, _k2 in predict_tokens:
            k1.append(_k1 + [0.0] * (max_tokens_len - len(_k1)))
            k2.append(_k2 + [0.0] * (max_tokens_len - len(_k2)))

        loss1 = categorical_crossentropy_loss(s1, k1)
        loss2 = categorical_crossentropy_loss(s2, k2)
        loss = loss1 + loss2

        return loss

    def on_epoch_end(self, epoch, logs=None):
        train_acc, wrong_train, train_loss = 0.0, [], 0.0  #self.evaluate(train_data)
        print(f"Evaluating dev data...")
        dev_data = self.bert_dev_data.data

        dev_loss, dev_acc, dev_recall, dev_f1, error_list = self.evaluate_by_ID(
            dev_data)
        #  dev_acc, wrong_eval, dev_loss = self.evaluate(dev_data)

        #  self.train_ACC.append(train_acc)
        self.dev_ACC.append(dev_acc)

        logger.info(f"Epoch {epoch+1}: "
                    f"dev_loss: {dev_loss:.4f}, "
                    f"dev_acc: {dev_acc:.4f}, "
                    f"dev_recall: {dev_recall:.4f}, "
                    f"dev_f1: {dev_f1:.4f}")

        # 保存loss最小的模型
        #  if dev_loss < self.best_loss:
        #      self.best_loss = dev_loss
        # 保存f1最大的模型
        if dev_f1 > self.best_f1:
            self.best_f1 = dev_f1

            best_model_weights_file = get_best_model_weights_file_name(
                selected_fold, Config.category)
            self.bert_model.train_model.save_weights(best_model_weights_file)
            logger.success(
                f"Saved best model weights to {best_model_weights_file}.")
            #  os.symlink()

            #  wrong_eval_file = get_wrong_eval_file_name(selected_fold)
            #  with open(wrong_eval_file, 'w') as F:
            #      last_ID = ""
            #      for ID, true_text, pred_text in error_list:
            #          if ID != last_ID:
            #              if last_ID:
            #                  F.write("\n")
            #              last_ID = ID
            #          F.write(f"{ID} | \t {true_text} | \t {pred_text}\n")

        #  if dev_loss > self.last_loss:
        #      self.num_loss_up += 1
        #      logger.warning(f"Dev loss keep raise {self.num_loss_up} times.")
        #  else:
        #      self.num_loss_up = 0
        if dev_f1 < self.last_f1:
            self.num_loss_up += 1
            logger.warning(f"Dev f1 keep decreased {self.num_loss_up} times.")
        else:
            self.num_loss_up = 0

        self.last_loss = dev_loss
        self.last_f1 = dev_f1
        #  print("")
    def evaluate_by_ID(self, data_list):
        dict_val_labels = {}
        test_data = []
        for text_in, c_in, label, ID in tqdm(iter(data_list)):
            if ID not in dict_val_labels:
                dict_val_labels[ID] = []
            dict_val_labels[ID].append(label)
            test_data.append((ID, text_in, c_in))

        from predict import extract_labels_from_test_data
        results = extract_labels_from_test_data(
            test_data,
            category=Config.category,
            top_k=Config.top_k,
            s_threshold=Config.s_threshold,
            e_threshold=Config.e_threshold,
            min_label_len=Config.min_label_len,
            seg_len=Config.seg_len,
            seg_backoff=Config.seg_backoff,
            bert_model=self.bert_model)

        #  print(results)
        dict_labels = {
            ID: [x[0] for x in items]
            for ID, items in results.items()
        }

        def text_similarity(s_true, s_pred, strictly_equal=True):
            if s_true == s_pred:
                return 1.0
            else:
                if strictly_equal:
                    return 0.0
                else:
                    if len(s_true) == 0:
                        return 0.0
                    s1_words = [w for w in s_true]
                    s2_words = [w for w in s_pred]
                    s1_end, s2_end, longest, substring = longest_common_substring(
                        s1_words, s2_words)
                    return longest / max(len(s1_words), len(s2_words))

        error_list = []
        loss = 0.0
        TP = FP = TN = FN = 0.0
        for ID, list_true_labels in dict_val_labels.items():
            tp = fp = tn = fn = 0.0
            list_pred_labels = dict_labels.get(ID, [])
            if list_pred_labels:

                # 严格按顺序对位比较
                #  for i, t_label in enumerate(list_true_labels):
                #      if i <= len(list_pred_labels) - 1:
                #          similarity_score = text_similarity(
                #              t_label,
                #              list_pred_labels[i],
                #              strictly_equal=self.strictly_equal)
                #          assert similarity_score >= 0.0 and similarity_score <= 1.0
                #          if similarity_score < 1.0:
                #              error_list.append(
                #                  (ID, t_label, list_pred_labels[i]))
                #
                #          tp += similarity_score
                #          fp += 1.0 - similarity_score
                #          loss += 1.0 - similarity_score
                #      else:
                #          fp += 1.0
                #          loss += 1.0
                #          error_list.append((ID, t_label, ""))
                #  if len(list_pred_labels) > len(list_true_labels):
                #      loss += (len(list_pred_labels) -
                #               len(list_true_labels)) * 1.0
                #      for pred_label in list_pred_labels[len(list_true_labels):]:
                #          error_list.append((ID, "", pred_label))

                # 取实际标注在预测标注的最大相似度
                for t_label in list_true_labels[:min(len(list_true_labels),
                                                     len(list_pred_labels))]:
                    max_similarity_score = 0.0
                    for p_label in list_pred_labels:
                        similarity_score = text_similarity(
                            t_label,
                            p_label,
                            strictly_equal=self.strictly_equal)
                        if similarity_score > max_similarity_score:
                            max_similarity_score = similarity_score
                    tp += max_similarity_score
                    fp += 1.0 - max_similarity_score
                    loss += 1.0 - max_similarity_score
                if len(list_pred_labels) > len(list_true_labels):
                    loss += (len(list_pred_labels) -
                             len(list_true_labels)) * 1.0
                    for pred_label in list_pred_labels[len(list_true_labels):]:
                        error_list.append((ID, "", pred_label))
                elif len(list_pred_labels) < len(list_true_labels):
                    loss += (len(list_true_labels) -
                             len(list_pred_labels)) * 1.0
                    fp += len(list_true_labels) - len(list_pred_labels)
                    for true_label in list_true_labels[len(list_pred_labels):]:
                        error_list.append((ID, true_label, ""))

            else:
                loss += len(list_true_labels) * 1.0
                fp += len(list_true_labels) * 1.0

            TP += tp
            FP += fp
            TN += tn
            FN += fn
        logger.debug(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        true_chars = sum(
            [len(labels) for ID, labels in dict_val_labels.items()])
        pred_chars = sum([len(labels) for ID, labels in dict_labels.items()])
        logger.debug(f"true_chars: {true_chars}, " f"pred_chars: {pred_chars}")
        #  if self.strictly_equal:
        #      assert true_chars == TP + FP

        acc = (TP + FN) / pred_chars
        recall = (TP + FN) / true_chars
        f1 = 2 * acc * recall / (acc + recall)

        return loss, acc, recall, f1, error_list

    def evaluate(self, data_list):
        A = 1e-10
        wrong_list = []
        i = 0
        tag_tokens = []
        for d in tqdm(iter(data_list)):
            R, tag_token = self.extract_entity(d[0], d[1])
            tag_tokens.append(tag_token)
            if R == d[2]:
                A += 1
            else:
                wrong_list.append((i, d[2], R))
            i += 1
        dev_loss = self.get_dev_loss(tag_tokens)
        return A / len(data_list), wrong_list, dev_loss


#%% ------------------------------------------------------------
def train_one_fold(bert_train_data,
                   bert_dev_data,
                   selected_fold,
                   force_run=False):
    print(f"Training fold {selected_fold}...")

    best_model_weights_file = get_best_model_weights_file_name(selected_fold, Config.category)
    if os.path.isfile(best_model_weights_file) or os.path.isfile(
            best_model_weights_file + ".index"):
        logger.info(
            f"The fold {selected_fold} has already been trained. Skip train step."
        )
        return

    from tf_utils import enable_gpu_growth
    enable_gpu_growth()

    train_data = bert_train_data.data
    dev_data = bert_dev_data.data

    additional_chars = set()
    for row in train_data + dev_data:
        additional_chars.update(
            re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', row[2]))
        #  "[^a-zA-Z0-9\/\.\,\?\(\)\[\]\{\}\u4e00-\u9fa5]"
    if u'，' in additional_chars:
        additional_chars.remove(u'，')

    bert_model = BertModel(Config.config_path, Config.checkpoint_path,
                           Config.dict_path, Config.learning_rate,
                           Config.max_seq_length)

    bert_model.build_model()

    def _extract_entity(text_in, c_in, bert_model, classes, additional_chars):
        return bert_model.extract_entity(text_in, c_in, additional_chars)

    def extract_entity_func():
        from functools import partial
        return partial(_extract_entity,
                       bert_model=bert_model,
                       classes=classes,
                       additional_chars=additional_chars)

    extract_entity = extract_entity_func()

    logger.info(f"Start train fold {selected_fold}...")
    evaluator = Evaluate(bert_dev_data, extract_entity, bert_model,
                         Config.learning_rate, Config.min_learning_rate,
                         Config.strictly_equal)
    #  es_monitor = 'val_loss'  # acc, loss, val_loss, val_acc
    #  es_patience = 50  # 当early stop被激活，则经过patience个epoch后停止训练。
    #  es_mode = 'min'  # 'auto', 'min', 'max'
    #  es_verbose = 2
    #  early_stopping = EarlyStopping(monitor=es_monitor,
    #                                 patience=es_patience,
    #                                 mode=es_mode,
    #                                 verbose=es_verbose)

    bert_model.train_model.fit_generator(
        bert_train_data.__iter__(Config.max_seq_length),
        steps_per_epoch=len(bert_train_data),
        #  epochs=3 * steps_per_update,
        epochs=Config.epochs,
        callbacks=[evaluator])

    bert_model = None
    K.clear_session()
    #  tf.reset_default_graph()


#%% ------------------------------------------------------------

if not ensure_empty_directory(Config.output_dir):
    exit(-1)
if not ensure_empty_directory(Config.results_dir):
    exit(-1)

#  batch_size = 24  # 16 for gpu@uubasin 32 for colab
batch_size = Config.batch_size
#  meminfo = get_gpu_info(gpu_id=0)
#  if meminfo.total <= 8 * 1000 * 1000 * 1000:
#      batch_size = 16
#  elif meminfo.total <= 12 * 1000 * 1000 * 1000:
#      batch_size = 24
#  else:
#      batch_size = 32
#  Config.batch_size = batch_size
print(f"batch_size: {batch_size}")

df_train = load_train_data(raw_train_data_file, Config.reverse_text)
classes = set(df_train.category.unique())
(random_order, cv_ids) = get_cross_validation_config(df_train,
                                                     Config.total_folds,
                                                     train_folds_file)
total_folds = len(cv_ids)

batch_folds = 1
folds = [total_folds - x - 1 for x in range(total_folds)]

logger.debug(f"folds: {folds}")
while len(folds) > 0:
    selected_folds = []
    for i in range(min(batch_folds, len(folds))):
        selected_folds.append(folds.pop())
    logger.debug(f"selected_folds: {selected_folds}")
    train_process = None
    for selected_fold in selected_folds:

        if Config.category == "WORKCONTENT":
            if selected_fold not in [2]:
                continue
        elif Config.category == 'PROJRESP':
            if selected_fold not in [0]:
                continue
        else:
            continue

        train_data, dev_data = get_train_dev_data(df_train, selected_fold,
                                                  random_order, cv_ids)
        train_data += dev_data

        tokenizer = BertTokenizer(Config.dict_path)
        bert_train_data = TrainData(train_data,
                                    tokenizer,
                                    Config.max_seq_length,
                                    batch_size=batch_size)
        bert_dev_data = TrainData(dev_data,
                                  tokenizer,
                                  Config.max_seq_length,
                                  batch_size=batch_size)
        logger.info(f"selected fold: {selected_fold}")
        logger.debug(
            f"len(train_data): {len(train_data)} len(dev_data): {len(dev_data)}"
        )

        train_process = Process(target=train_one_fold,
                                args=(bert_train_data, bert_dev_data,
                                      selected_fold, False))
        train_process.start()
    if train_process:
        train_process.join()
