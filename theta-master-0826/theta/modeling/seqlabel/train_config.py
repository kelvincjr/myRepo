#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Config:

    #  def update(cfg):
    #      for a in cfg.__dict__:
    #          if not a.startswith('_'):
    #              cfg[a]

    SEED = 8864
    debug_extract_labels = False

    surfix = ""

    experiment = "resumes"
    data_dir = f"../user_data/seqlabel_datasets/{experiment}"
    results_dir = f"../prediction_result/{experiment}_result/seqlabel"
    output_dir = f"../user_data/{experiment}_output/seqlabel"
    train_data_json_file = f"../data/Dataset/train_data/train_data.json"

    category = ""
    top_k = 20
    s_threshold = 0.1
    e_threshold = 0.1  #0.1
    min_label_len = 8
    seg_len = 510
    seg_backoff = 400
    use_additional_section = True
    strictly_equal = False
    num_augments = 0
    surfix = f"_aug{num_augments}"

    # -------------------- seqlabel model train & test begin. ---------------
    total_folds = 5

    # 是否应该取训练数据最大长度或最大长度95%（99%）？
    max_seq_length = seg_len  # 510  #160

    # -------------------- seqlabel model train & test end. ---------------

    BERT_CHINESE_DIR = "../data/External/pretrained/tensorflow/bert-base-chinese"
    epochs = 6  # 训练轮次
    batch_size = 8  # 按GPU显存大小调整，建议8G 取16, 12G 取24，16G 取32

    learning_rate = 5e-5
    min_learning_rate = 1e-5
    #  learning_rate = 1e-5
    #  min_learning_rate = 1e-6

    # 反序文本和标签，训练新模型
    reverse_text = False
    # ls -l /dev/nvidia? | wc -l
    gpus = 1

    # TPU
    steps_per_update = 8  # 权重更新前梯度累积的步数。
    #  learning_rate = 5e-4  # 运用梯度累积时，适当调高学习率

    config_path = f"{BERT_CHINESE_DIR}/bert_config.json"
    checkpoint_path = f"{BERT_CHINESE_DIR}/bert_model.ckpt"
    dict_path = f"{BERT_CHINESE_DIR}/vocab.txt"


class Config_WORKCONTENT(Config):
    category = "WORKCONTENT"
    top_k = 20
    s_threshold = 0.1
    e_threshold = 0.01
    min_label_len = 8
    seg_len = 510
    seg_backoff = 400
    strictly_equal = False
    use_additional_section = True
    num_augments = 0
    surfix = f"_aug{num_augments}"


class Config_PROJRESP(Config):
    category = "PROJRESP"
    top_k = 20
    s_threshold = 0.1
    e_threshold = 0.1  #0.1
    min_label_len = 8
    seg_len = 510
    seg_backoff = 400
    use_additional_section = True
    strictly_equal = False
    num_augments = 0
    surfix = f"_aug{num_augments}"


def get_best_model_file_name(selected_fold: int) -> str:
    return f"{Config.output_dir}/best_model_fold{selected_fold}{surfix}.model"


def get_best_model_weights_file_name(selected_fold: int, category) -> str:
    return f"{Config.output_dir}/best_model_{category}_fold{selected_fold}{Config.surfix}.weights"


def get_result_file_name(selected_fold: int) -> str:
    return f"{Config.results_dir}/result_{Config.category}_fold{selected_fold}{Config.surfix}.txt"


def get_wrong_eval_file_name(selected_fold: int) -> str:
    return f"{Config.output_dir}/wrong_eval_fold{selected_fold}{Config.surfix}.txt"
