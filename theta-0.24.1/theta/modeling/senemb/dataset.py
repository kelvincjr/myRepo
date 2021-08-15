#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import json
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from ...utils.multiprocesses import barrier_leader_process, barrier_member_processes, is_master_process, is_multi_processes


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


class InputFeatures(object):
    def __init__(self, input_ids_a, attention_mask_a, input_ids_b,
                 attention_mask_b, label_id):
        self.input_ids_a = input_ids_a
        self.attention_mask_a = attention_mask_a
        self.input_ids_b = input_ids_b
        self.attention_mask_b = attention_mask_b
        self.label_id = label_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(
    examples,
    #  glue_labels,
    label2id,
    tokenizer,
    max_length=512,
    task=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        glue_labels: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    #  label2id = {label: i for i, label in enumerate(glue_labels)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        def example_to_feature(text_a, text_b):
            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                return_token_type_ids=True,
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs[
                "token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0
                              ] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] *
                                  padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] *
                                  padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] *
                                                   padding_length)

            assert len(
                input_ids
            ) == max_length, "Error with input length {} vs {}".format(
                len(input_ids), max_length)
            assert len(
                attention_mask
            ) == max_length, "Error with input length {} vs {}".format(
                len(attention_mask), max_length)
            assert len(
                token_type_ids
            ) == max_length, "Error with input length {} vs {}".format(
                len(token_type_ids), max_length)

            return input_ids, attention_mask, token_type_ids

        input_ids_a, attention_mask_a, token_type_ids_a = example_to_feature(
            example.text_a, None)
        input_ids_b, attention_mask_b, token_type_ids_b = example_to_feature(
            example.text_b, None)

        label = 0
        if example.label is not None:
            if output_mode == "classification":
                label = label2id[example.label]
            elif output_mode == "regression":
                label = float(example.label)
            else:
                raise KeyError(output_mode)

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))

            tokens_a = tokenizer.tokenize(example.text_a)
            logger.info(f"tokens_a: {' '.join([str(x) for x in tokens_a])}")

            logger.info("input_ids_a: %s" %
                        " ".join([str(x) for x in input_ids_a]))
            logger.info("attention_mask_a: %s" %
                        " ".join([str(x) for x in attention_mask_a]))
            logger.info("token_type_ids_a: %s" %
                        " ".join([str(x) for x in token_type_ids_a]))

            tokens_b = tokenizer.tokenize(example.text_b)
            logger.info(f"tokens_b: {' '.join([str(x) for x in tokens_b])}")
            logger.info("input_ids_b: %s" %
                        " ".join([str(x) for x in input_ids_b]))
            logger.info("attention_mask_b: %s" %
                        " ".join([str(x) for x in attention_mask_b]))
            logger.info("token_type_ids_b: %s" %
                        " ".join([str(x) for x in token_type_ids_b]))

            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(input_ids_a, attention_mask_a, input_ids_b,
                          attention_mask_b, label))

    return features


def examples_to_dataset(args,
                        examples,
                        glue_labels,
                        tokenizer,
                        max_seq_length,
                        evaluate=False,
                        output_mode="classification",
                        alias=""):
    # Make sure only the first process in distributed training process the dataset,
    # and the others will use the cache
    if not evaluate:
        barrier_member_processes(args)

    args.id2label = {i: label for i, label in enumerate(glue_labels)}
    args.label2id = {label: i for i, label in enumerate(glue_labels)}
    args.num_labels = len(glue_labels)

    # Load data features from cache or dataset file
    cached_features_file = Path(
        args.data_dir
    ) / f"cached_glue_{args.dataset_name}{'_' + alias if alias else ''}"
    if args.cache_features and cached_features_file.exists(
    ) and not args.overwrite_cache:
        logger.info(
            f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        logger.info(f"Creating features from dataset file at {args.data_dir}")

        features = convert_examples_to_features(
            examples,
            args.label2id,
            tokenizer,
            max_length=max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(
                args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
        )

        if args.cache_features and is_master_process(args):
            logger.info(
                f"Saving features into cached file {cached_features_file}")
            torch.save(features, cached_features_file)

    logger.info(
        f"convert {len(examples)} examples to {len(features)} features.")
    # Make sure only the first process in distributed training process the dataset,
    # and the others will use the cache
    if not evaluate:
        barrier_leader_process(args)

    # Convert to Tensors and build dataset
    all_input_ids_a = torch.tensor([f.input_ids_a for f in features],
                                   dtype=torch.long)
    all_attention_mask_a = torch.tensor([f.attention_mask_a for f in features],
                                        dtype=torch.long)
    all_input_ids_b = torch.tensor([f.input_ids_b for f in features],
                                   dtype=torch.long)
    all_attention_mask_b = torch.tensor([f.attention_mask_b for f in features],
                                        dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features],
                                     dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features],
                                     dtype=torch.float)

    dataset = TensorDataset(all_input_ids_a, all_attention_mask_a,
                            all_input_ids_b, all_attention_mask_b,
                            all_label_ids)
    return dataset


def build_encode_dataset(args, examples, tokenizer, max_seq_length):

    all_input_ids = []
    all_attention_mask = []
    for example in examples:
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_seq_length,
            return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs[
            "token_type_ids"]
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        pad_token = tokenizer.pad_token_id
        pad_token_segment_id = tokenizer.pad_token_type_id,
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] *
                                           padding_length)
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask)
    return dataset
