#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random, copy, json
from tqdm import tqdm
from loguru import logger
from theta.utils import seg_generator
from dataclasses import dataclass, field
import numpy as np
#  import mlflow

from ..utils import seg_generator


#  @dataclass(frozen=False)
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    #  guid: str
    #  text_a: str
    #  text_b: Optional[str] = None
    #  label: Optional[str] = None
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2,
                          sort_keys=True) + "\n"


def load_glue_examples(data_generator, examples_file):

    examples = []

    for guid, text_a, text_b, label in data_generator(examples_file):
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    logger.info(f"Loaded {len(examples)} examples.")

    return examples


def load_train_val_examples(args,
                            train_data_generator,
                            glue_labels,
                            shuffle=True,
                            train_rate=0.9,
                            num_augments=0):
    all_train_examples = load_glue_examples(train_data_generator,
                                            args.train_file)

    if args.train_sample_rate < 1.0:
        num_samples = int(len(all_train_examples) * args.train_sample_rate)
        all_train_examples = all_train_examples[:num_samples]

    # 切分训练集和验证集
    # theta 提供split_train_eval_examples辅助函数
    from theta.utils import split_train_eval_examples
    train_examples, val_examples = split_train_eval_examples(
        all_train_examples,
        train_rate=args.train_rate,
        fold=args.fold,
        shuffle=shuffle)

    #  random.shuffle(all_train_examples)
    #  num_train_examples = int(len(all_train_examples) * args.train_rate)
    #  val_examples = all_train_examples[num_train_examples:]
    #  train_examples = all_train_examples[:num_train_examples]

    logger.info(f"Loaded {len(train_examples)} train examples, "
                f"{len(val_examples)} val examples.")
    return train_examples, val_examples


def load_test_examples(args, test_data_generator):
    test_base_examples = load_glue_examples(test_data_generator,
                                            args.test_file)

    logger.info(f"Loaded {len(test_base_examples)} test examples.")
    return test_base_examples


def show_glue_datainfo(glue_labels, train_data_generator, train_file,
                       test_data_generator, test_file):
    train_lengths = [
        len(text_a)
        for guid, text_a, _, label in train_data_generator(train_file)
    ]

    all_labels = []
    for _, _, _, label in train_data_generator(train_file):
        all_labels.append(label)
        if label not in glue_labels:
            glue_labels.append(label)

    test_lengths = [
        len(text_a)
        for guid, text_a, _, label in test_data_generator(test_file)
    ]

    logger.info(f"****** glue_labels ******")
    logger.info(f"{glue_labels}")
    if isinstance(all_labels[0], str):
        from collections import Counter
        logger.info(f"{Counter(all_labels).most_common()}")
    logger.info(f"")

    logger.info(f"Train samples: {len(train_lengths)}")
    logger.info(f"Test samples: {len(test_lengths)}")

    logger.info(f"")
    import numpy as np
    logger.info(f"****** train lengths ******")
    logger.info(f"mean: {np.mean(train_lengths):.2f}")
    logger.info(f"std: {np.std(train_lengths):.2f}")
    logger.info(f"min: {np.min(train_lengths)}")
    logger.info(f"max: {np.max(train_lengths)}")
    logger.info(f"")

    logger.info(f"****** test lengths ******")
    logger.info(f"mean: {np.mean(test_lengths):.2f}")
    logger.info(f"std: {np.std(test_lengths):.2f}")
    logger.info(f"min: {np.min(test_lengths)}")
    logger.info(f"max: {np.max(test_lengths)}")
    logger.info(f"")


def save_glue_preds(args, preds, test_examples, probs=None):
    assert len(test_examples) == len(preds)
    reviews_file = args.reviews_file

    reviews = {}
    for i, (input_example, v) in enumerate(zip(test_examples, preds)):
        guid = input_example.guid
        text_a = input_example.text_a or ""
        text_b = input_example.text_b or ""
        if isinstance(v, list) or isinstance(v, np.ndarray):
            label = [args.id2label[i] for i, x in enumerate(v) if x]
        else:
            label = args.id2label[v]

        prob = None
        if isinstance(probs, np.ndarray):
            prob = [f"{x:.3f}" for x in list(probs[i])]
        elif isinstance(probs, list):
            prob = [f"{x:.3f}" for x in probs[i]]
        else:
            prob = None

        reviews[guid] = {
            'guid': guid,
            'text_a': text_a,
            'text_b': text_b,
            'label': label,
            'probs': prob
        }

    json.dump(reviews, open(reviews_file, 'w'), ensure_ascii=False, indent=2)
    #  with open(reviews_file, 'w') as fw:
    #      fw.write("guid\ttext_a\ttext_b\tlabel\n")
    #      for input_example, v in zip(test_examples, preds):
    #          guid = input_example.guid
    #          text_a = input_example.text_a or ""
    #          text_b = input_example.text_b or ""
    #          label = args.id2label[v]
    #          fw.write(f"{guid}\t{text_a}\t{text_b}\t{label}\n")

    logger.info(f"Total {len(preds)} lines saved to {reviews_file}")

    #  ----- Tracking -----
    #  if args.do_experiment:
    #      logger.debug(f"mlflow tracking uri: {mlflow.get_tracking_uri()}")
    #      mlflow.log_param(f"{args.dataset_name}_reviews_file", reviews_file)
    #      mlflow.log_artifact(reviews_file, args.artifact_path)

    return reviews_file
