# -*- coding: utf-8 -*-
import os
import re, pickle
import pandas as pd
import numpy as np
from loguru import logger


def determine_sep(filename):
    sep = ','
    idx0 = filename.rfind('.')
    if idx0 > 0:
        if filename[idx0 + 1:] == 'tsv':
            sep = '\t'
            logger.debug(f"Determine sep is '\\t', filename: {filename}")
    return sep


def load_train_data(raw_train_data_file: str,
                    reverse_text: bool = False) -> pd.DataFrame:
    sep = determine_sep(raw_train_data_file)

    df_train = pd.read_csv(raw_train_data_file,
                           names=['ID', 'text', 'category', 'subject'],
                           sep=sep,
                           header=None)
    print(f"df_train.shape: {df_train.shape}")
    df_train = df_train[df_train.category != "其他"]

    #  df_train.text = df_train.text.map(
    #      lambda x: re.sub("[^\w\*\-\+\.\(\)]", " ", x))
    #  df_train.text = df_train.text.map(
    #      lambda x: re.sub("[^\w\*\-\+\.\(\)\（\）]", " ", x))

    if reverse_text:
        df_train.text = df_train.text.map(lambda x: x[::-1])
        df_train.subject = df_train.subject.map(lambda x: x[::-1]
                                                if type(x) == str else "NaN")

    return df_train


def get_train_data_list(df_data: pd.DataFrame) -> list:
    return [(t, c, s, i) for t, c, s, i in zip(df_data.text, df_data.category,
                                               df_data.subject, df_data.ID)]


def load_test_data(raw_test_data_file: str,
                   reverse_text: bool = False) -> pd.DataFrame:
    sep = determine_sep(raw_test_data_file)
    df_test = pd.read_csv(raw_test_data_file,
                          names=['ID', 'text', 'category', 'subject'],
                          sep=sep,
                          header=None)
    df_test.text = df_test.text.fillna("")
    #  df_test.text = df_test.text.map(
    #      lambda x: re.sub("[^\w\*\-\+\.\(\)\（\）]", " ", x))
    if reverse_text:
        df_test.text = df_test.text.map(lambda x: x[::-1])

    return df_test


def get_test_data_list(df_test):
    return [(i, t, c)
            for i, t, c in zip(df_test.ID, df_test.text, df_test.category)]


def save_cross_validation_config(df_train, n_splits, filename):
    random_order = [x for x in range(df_train.shape[0])]
    #  np.random.shuffle(random_order)
    random_order = np.random.permutation(random_order)

    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()
    print(f"{df_train['category'].unique()}")
    y = label_encoder.fit_transform(df_train['category'])

    def cross_validation(df_train, y, n_splits=5):
        from sklearn.model_selection import StratifiedKFold
        cv_ids = []
        kf = StratifiedKFold(n_splits=n_splits)
        for train, test in kf.split(df_train, y):
            cv_ids.append((train, test))
        return cv_ids

    cv_ids = cross_validation(df_train, y, n_splits=n_splits)

    # Save cross validation ids

    with open(filename, 'wb') as F:
        pickle.dump((random_order, cv_ids), F)

    return random_order, cv_ids


def load_cross_validation_config(filename):
    with open(filename, 'rb') as F:
        random_order, cv_ids = pickle.load(F)
    return random_order, cv_ids


def get_cross_validation_config(df_train, total_folds, train_folds_file):
    #  if os.path.isfile(train_folds_file):
    #      random_order, cv_ids = load_cross_validation_config(
    #          filename=train_folds_file)
    #  else:
    random_order, cv_ids = save_cross_validation_config(
        df_train, n_splits=total_folds, filename=train_folds_file)
    return (random_order, cv_ids)


def get_train_dev_data(df_train, selected_fold, random_order, cv_ids):
    train_ids, test_ids = cv_ids[selected_fold]

    df_train_random = df_train.iloc[random_order, :]

    df_train_data = df_train_random.iloc[train_ids, :]
    df_dev_data = df_train_random.iloc[test_ids, :]

    train_data = get_train_data_list(df_train_data)
    dev_data = get_train_data_list(df_dev_data)

    # 交换训练集与验证集，用小的数据集训练，去预测大的数据集

    #  dev_order = [x for x in range(df_dev_data.shape[0])]
    #  np.random.shuffle(dev_order)
    #  df_dev_random = df_train.iloc[dev_order, :]
    #
    #  dev_data = get_train_data_list(df_dev_random)
    #  train_data = get_train_data_list(df_dev_data)

    #  train_data = [
    #      (t, c, s, i)
    #      for t, c, s, i in zip(df_train_data.text, df_train_data.category,
    #                            df_train_data.subject, df_train_data.ID)
    #  ]
    #  dev_data = [(t, c, s, i)
    #              for t, c, s, i in zip(df_dev_data.text, df_dev_data.category,
    #                                    df_dev_data.subject, df_dev_data.ID)]
    return train_data, dev_data
