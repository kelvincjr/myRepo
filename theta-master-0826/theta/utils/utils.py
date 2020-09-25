#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, random, json
import numpy as np
from tqdm import tqdm
from loguru import logger
import math
import torch
from sklearn.metrics import f1_score
from .multiprocesses import is_single_process


def load_pytorch_model(model, model_path):
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path,
                                                map_location='cpu'),
                                     strict=False)
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'),
                              strict=False)
    return model


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / e_x.sum(axis=1).reshape(-1, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    correct = np.sum((preds == labels).astype(int))
    acc = correct / preds.shape[0]
    f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
    acc_and_f1 = (acc + f1) / 2
    return {"acc": acc, "f1": f1, "acc_and_f1": acc_and_f1}


def init_random(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Setup CUDA, GPU & distributed training
def init_cuda(args):
    #  if args.local_rank == -1 or args.no_cuda:
    if is_single_process(args) or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will
        # take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    args.device = device


def init_theta(args):
    init_random(args.seed)
    init_cuda(args)


# DataFrame直接初始化出InputExample列表
def dataframe_to_examples(df_data, InputExample, cols_map=None):
    if cols_map is None:
        cols_map = {x: x for x in df_data.columns()}
    examples = []
    for i, row in df_data.iterrows():
        values = {k: str(row[v]) for k, v in cols_map.items()}
        examples.append(InputExample(**values))

    return np.array(examples)


def get_list_size(samples: [list, np.array]):
    if isinstance(samples, list):
        num_samples = len(samples)
    elif isinstance(samples, np.ndarray):
        num_samples = samples.shape[0]
    else:
        raise TypeError(
            f"Argument samples in shuffle_list() must be a list or np.ndarray."
        )
    return num_samples


def shuffle_list(samples: [list, np.array],
                 random_state=None) -> [list, np.array]:
    if random_state:
        np.random.seed(random_state)

    num_samples = get_list_size(samples)
    indices = np.random.randint(0, num_samples, num_samples)
    shuffled_samples = [samples[i] for i in indices]

    if isinstance(samples, np.ndarray):
        return np.array(shuffled_samples)
    else:
        return shuffled_samples


def list_to_list(samples: [list, np.array]) -> list:
    if isinstance(samples, np.ndarray):
        samples = samples.tolist()
        return samples
    elif isinstance(samples, list):
        return samples
    else:
        raise TypeError(
            f"Samples in list_to_list() must be a list or np.ndarray.")


def concatenate_list(a_samples: [list, np.array],
                     b_samples: [list, np.array]) -> [list, np.array]:
    if isinstance(a_samples, np.ndarray):
        a_samples = list_to_list(a_samples)
        b_samples = list_to_list(b_samples)
        samples = a_samples + b_samples
        return np.array(samples)
    elif isinstance(a_samples, list):
        a_samples = list_to_list(a_samples)
        b_samples = list_to_list(b_samples)
        samples = a_samples + b_samples
        return samples
    else:
        raise TypeError(
            f"Samples in concatenote_list() must be a list or np.ndarray.")


def to_numpy(X):
    """
    Convert input to numpy ndarray
    """
    if hasattr(X, 'iloc'):  # pandas
        return X.values
    elif isinstance(X, list):  # list
        return np.array(X)
    elif isinstance(X, np.ndarray):  # ndarray
        return X
    else:
        raise ValueError("Unable to handle input type %s" % str(type(X)))


def unpack_text_pairs(X):
    assert isinstance(X, list) or isinstance(X, np.ndarray)
    return (X, None) if X.ndim == 1 else (X[:, 0], X[:, 1])


def unpack_data(X, y=None):
    return unpack_text_pairs(to_numpy(X)), to_numpy(y) if y else None


def create_logger(log_dir: str, logger_name: str):
    # 创建一个logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # 创建一个handler，
    timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
    log_filename = f"{log_dir}/log_{logger_name}_{timestamp}.txt"
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s] ## %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def seg_generator(iterables, seg_len, seg_backoff=0):
    if seg_len <= 0:
        yield iterables, 0
    else:
        #  # 确保iterables列表中每一项的条目数相同
        #  assert sum([len(x)
        #              for x in iterables]) == len(iterables[0]) * len(iterables)
        assert iterables[0] is not None
        s0 = 0
        while s0 < len(iterables[0]):
            s1 = s0 + seg_len
            segs = [x[s0:s1] if x else None for x in iterables]
            yield segs, s0
            s0 += seg_len - seg_backoff


class slide_generator:
    def __init__(self, words: list, slide_len, slide_backoff=0):

        assert slide_offset <= slide_len

        self.words = words
        self.slide_len = slide_len
        self.slide_backoff = slide_backoff

        num_words = len(words)
        self.total_slides = num_words // (slide_len - slide_backoff) + 1

    def __iter__(self):
        for i in range(self.total_slides):
            yield self.__getitem__(i)

    def __len__(self):
        return self.total_slides

    def __getitem__(self, idx):
        s = (self.slide_len - self.slide_backoff) * idx
        e = (self.slide_len - self.slide_backoff) * (idx + 1)
        slide_words = self.words[s:e]

        return slide_words


def split_train_eval_examples(examples: [list, np.array],
                              train_rate=0.9,
                              fold=0,
                              shuffle=False,
                              random_type=None):
    examples = list(examples)
    num_examples = len(examples)

    # for medical_entity
    if random_type == "np":
        if shuffle:
            examples = shuffle_list(examples, random_state=None)

        num_eval_examples = int(num_examples * (1 - train_rate))
        assert fold <= num_examples // num_eval_examples

        s = num_eval_examples * fold
        e = num_eval_examples * (fold + 1)
        eval_examples = examples[s:e]
        train_examples = concatenate_list(examples[:s], examples[e:])
    else:
        # for entity_typing
        if shuffle:
            random.shuffle(examples)

        num_train_examples = int(num_examples * train_rate)
        num_eval_examples = num_examples - num_train_examples

        e = num_examples - num_eval_examples * fold
        s = num_examples - num_eval_examples * (fold + 1)
        if s < 0:
            s = 0
        eval_examples = examples[s:e]
        train_examples = concatenate_list(examples[:s], examples[e:])

    return train_examples, eval_examples


class batch_generator:
    def __init__(self,
                 examples: [list, np.array],
                 train_rate=0.9,
                 shuffle=False,
                 random_state=None):
        if shuffle:
            examples = shuffle_list(examples, random_state=random_state)
        num_examples = get_list_size(examples)
        num_eval_examples = int(num_examples * (1 - train_rate))

        self.examples = examples
        self.total_batchs = num_examples // num_eval_examples + 1

    def __iter__(self):
        for i in range(self.total_batchs):
            yield self.__getitem__(i)

    def __getitem__(self, idx):
        s = self.num_eval_examples * idx
        e = self.num_eval_examples * (idx + 1)
        eval_examples = self.examples[s:e]
        train_examples = concatenate_list(self.examples[:s], self.examples[e:])
        return train_examples, eval_examples

    def __len__(self):
        return self.total_batchs


def get_pred_results_file(args):
    return f"{args.output_dir}/{args.dataset_name}_result.json"


def get_submission_file(args):
    return f"{args.output_dir}/{args.dataset_name}_submission.json"


def load_json_file(json_file):
    rd = open(json_file, 'r')
    lines = rd.readlines()
    rd.close()
    json_data = []
    for line in tqdm(lines):
        line = line.strip()
        line_data = json.loads(line)
        json_data.append(line_data)
    print(f"Total: {len(json_data)}")
    print(json_data[:5])
    return json_data


def show_statistical_distribution(data_list):
    logger.info(f"***** Statistical Distribution *****")
    logger.info(f"mean: {np.mean(data_list):.4f}")
    logger.info(f"std: {np.mean(data_list):.4f}")
    logger.info(f"max: {np.max(data_list):.4f}")
    logger.info(f"min: {np.min(data_list)}:.4f")
