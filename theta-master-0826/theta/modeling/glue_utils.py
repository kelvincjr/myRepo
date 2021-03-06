#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random, copy, json
from tqdm import tqdm
from loguru import logger
from theta.utils import seg_generator
from dataclasses import dataclass, field
#import mlflow

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
    from collections import Counter
    logger.info(f"{Counter(all_labels).most_common()}")
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


def save_glue_preds(args, preds, test_examples):
    assert len(test_examples) == len(preds)
    reviews_file = f"{args.latest_dir}/{args.dataset_name}_reviews_{args.local_id}.tsv"
    with open(reviews_file, 'w') as fw:
        fw.write("guid\ttext_a\ttext_b\tlabel\n")
        for input_example, v in zip(test_examples, preds):
            guid = input_example.guid
            text_a = input_example.text_a or ""
            text_b = input_example.text_b or ""
            label = args.id2label[v]
            fw.write(f"{guid}\t{text_a}\t{text_b}\t{label}\n")
    logger.info(f"Total {len(preds)} lines saved to {reviews_file}")

    #  ----- Tracking -----
    if args.do_experiment:
        logger.debug(f"mlflow tracking uri: {mlflow.get_tracking_uri()}")
        mlflow.log_param(f"{args.dataset_name}_reviews_file", reviews_file)
        mlflow.log_artifact(reviews_file, args.artifact_path)

    return reviews_file
