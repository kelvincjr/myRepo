#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, copy
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from ...utils.multiprocesses import barrier_leader_process, barrier_member_processes, is_multi_processes
from ...utils import seg_generator
from ..ner_utils import InputExample


class InputFeature(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len, segment_ids,
                 start_ids, end_ids, subjects):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_ids = start_ids
        self.input_len = input_len
        self.end_ids = end_ids
        self.subjects = subjects

    def __repr__(self):
        return str(self.to_json_string())

    def __eq__(self, other):
        return self.input_ids == other.input_ids and \
            self.input_mask == other.input_mask and \
            self.segment_ids == other.segment_ids and \
            self.start_ids == other.start_ids and \
            self.end_ids == other.end_ids and \
            self.subjects == other.subjects and \
            self.input_len == other.input_len

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


#  def load_examples(args,
#                    data_generator,
#                    examples_file,
#                    seg_len=0,
#                    seg_backoff=0):
#
#      examples = []
#
#      for guid, text_a, text_b, labels in data_generator(
#              args, examples_file, seg_len=seg_len, seg_backoff=seg_backoff):
#          assert text_a is not None
#          #  for (seg_text_a, seg_text_b) in seg_generator((text_a, text_b),
#          #                                                seg_len, seg_backoff):
#          #      seg_words_a = [w for w in seg_text_a]
#          #
#          #      examples.append(
#          #          InputExample(guid=guid, text_a=seg_words_a, labels=labels))
#          examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
#      logger.info(f"Loaded {len(examples)} examples.")
#
#      return examples


def encode_examples(examples, label2id, tokenizer, max_seq_length):

    #  texts = [''.join(e.text_a) for e in examples]
    texts = [e.text_a[:max_seq_length - 2] for e in examples]

    #  outputs = tokenizer.batch_encode_plus(texts,
    #                                        max_length=max_seq_length,
    #                                        add_special_tokens=True,
    #                                        return_tensors='pt',
    #                                        pad_to_max_length=True,
    #                                        return_attention_mask=True,
    #                                        return_token_type_ids=True)
    #  all_input_ids, all_attention_mask, all_token_type_ids = outputs[
    #      'input_ids'], outputs['attention_mask'], outputs['token_type_ids']

    all_tokens = [[tokenizer.cls_token] + tokenizer.tokenize(text) +
                  [tokenizer.sep_token]
                  for text in tqdm(texts, desc="Tokenize")]
    all_input_lens = [len(tokens) for tokens in all_tokens]
    all_input_ids = [
        tokenizer.convert_tokens_to_ids(tokens) for tokens in all_tokens
    ]
    all_attention_mask = [[1] * len(input_ids) for input_ids in all_input_ids]
    all_token_type_ids = [[0] * len(input_ids) for input_ids in all_input_ids]

    all_padding_lens = [max_seq_length - n for n in all_input_lens]
    for i, (input_ids, attention_mask, token_type_ids,
            padding_length) in enumerate(
                zip(all_input_ids, all_attention_mask, all_token_type_ids,
                    all_padding_lens)):
        all_input_ids[i] = input_ids + [0] * padding_length
        all_attention_mask[i] = attention_mask + [0] * padding_length
        all_token_type_ids[i] = token_type_ids + [0] * padding_length

    def encode_subjects(subjects, num_tokens):
        #  logger.warning(f"num_tokens: {num_tokens}")
        start_ids = [label2id['[unused1]']] * num_tokens
        end_ids = [label2id['[unused1]']] * num_tokens
        subjects_id = []
        if subjects:
            for subject in subjects:
                #  logger.debug(f"subject: {subject}")
                label = subject[0]
                start = subject[1]
                end = subject[2]
                start_ids[start + 1] = label2id[label]
                end_ids[end + 1] = label2id[label]
                subjects_id.append((label2id[label], start, end))
        return subjects_id, start_ids, end_ids

    all_subjects_ids = []
    all_start_ids = []
    all_end_ids = []
    all_subjects = [e.labels for e in examples]
    for input_ids, attention_mask, token_type_ids, subjects in zip(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_subjects):
        num_tokens = len(input_ids)
        subjects_id, start_ids, end_ids = encode_subjects(subjects, num_tokens)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length

        all_subjects_ids.append(subjects_id)
        all_start_ids.append(start_ids)
        all_end_ids.append(end_ids)

    #  all_start_ids = torch.tensor(all_start_ids, dtype=torch.long)
    #  all_end_ids = torch.tensor(all_end_ids, dtype=torch.long)
    #  all_input_lens = torch.tensor(all_input_lens, dtype=torch.long)

    #  all_input_lens = torch.tensor(
    #      [len(input_ids) for input_ids in all_input_ids], dtype=torch.long)

    logger.debug(f"all_input_ids.shape: {np.array(all_input_ids).shape}")
    logger.debug(
        f"all_attention_mask.shape: {np.array(all_attention_mask).shape}")
    logger.debug(
        f"all_token_type_ids.shape: {np.array(all_token_type_ids).shape}")
    logger.debug(f"all_start_ids.shape: {np.array(all_start_ids).shape}")
    logger.debug(f"all_end_ids.shape: {np.array(all_end_ids).shape}")
    logger.debug(f"all_subjects_ids.shape: {np.array(all_subjects_ids).shape}")
    logger.debug(f"all_input_lens.shape: {np.array(all_input_lens).shape}")
    assert np.array(all_input_ids).shape[1] == max_seq_length
    assert np.array(all_attention_mask).shape[1] == max_seq_length
    assert np.array(all_token_type_ids).shape[1] == max_seq_length

    all_features = [
        InputFeature(input_ids=input_ids,
                     input_mask=attention_mask,
                     segment_ids=token_type_ids,
                     start_ids=start_ids,
                     end_ids=end_ids,
                     subjects=subjects_ids,
                     input_len=input_len) for input_ids, attention_mask,
        token_type_ids, start_ids, end_ids, subjects_ids, input_len in zip(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_start_ids, all_end_ids, all_subjects_ids, all_input_lens)
    ]
    return (all_features, {
        'input_ids':
        torch.tensor(all_input_ids, dtype=torch.long),
        'attention_mask':
        torch.tensor(all_attention_mask, dtype=torch.long),
        'token_type_ids':
        torch.tensor(all_token_type_ids, dtype=torch.long),
        'start_ids':
        torch.tensor(all_start_ids, dtype=torch.long),
        'end_ids':
        torch.tensor(all_end_ids, dtype=torch.long),
        "subjects_ids":
        all_subjects_ids,
        "input_lens":
        torch.tensor(all_input_lens, dtype=torch.long),
    })


def examples_to_dataset(examples, label2id, tokenizer, max_seq_length):
    all_features, outputs = encode_examples(examples, label2id, tokenizer,
                                            max_seq_length)

    #  dataset_old, features_old = examples_to_dataset_old(
    #      examples, label2id, tokenizer, max_seq_length)
    #
    #  for x, x1 in zip(all_features, features_old):
    #      if x != x1:
    #          logger.info(f"input_ids: {len(x.input_ids)}, {x.input_ids}")
    #          logger.info(f"input_mask: {len(x.input_mask)}, {x.input_mask}")
    #          logger.info(f"segment_ids: {len(x.segment_ids)}, {x.segment_ids}")
    #          logger.info(f"start_ids: {len(x.start_ids)}, {x.start_ids}")
    #          logger.info(f"end_ids: {len(x.end_ids)}, {x.end_ids}")
    #          logger.info(f"subjects: {x.subjects}")
    #          logger.info(f"input_len: {x.input_len}")
    #
    #          if x.input_ids != x1.input_ids:
    #              logger.warning(
    #                  f"input_ids: {len(x1.input_ids)}, {x1.input_ids}")
    #          if x.input_mask != x1.input_mask:
    #              logger.warning(
    #                  f"input_mask: {len(x1.input_mask)}, {x1.input_mask}")
    #          if x.segment_ids != x1.segment_ids:
    #              logger.warning(
    #                  f"segment_ids: {len(x1.segment_ids)}, {x1.segment_ids}")
    #          if x.start_ids != x1.start_ids:
    #              logger.warning(
    #                  f"start_ids: {len(x1.start_ids)}, {x1.start_ids}")
    #          if x.end_ids != x1.end_ids:
    #              logger.warning(f"end_ids: {len(x1.end_ids)}, {x1.end_ids}")
    #          if x.subjects != x1.subjects:
    #              logger.warning(f"subjects: {x1.subjects}")
    #          if x.input_len != x1.input_len:
    #              logger.warning(f"input_len: {x1.input_len}")

    dataset = TensorDataset(
        outputs['input_ids'],
        outputs['attention_mask'],
        outputs['token_type_ids'],
        outputs['start_ids'],
        outputs['end_ids'],
        outputs['input_lens'],
    )

    assert len(dataset) == len(all_features)

    return dataset, all_features


def convert_examples_to_features(
    examples,
    label2id,
    tokenizer,
    max_seq_length,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for ex_index, e in enumerate(tqdm(examples, desc="Convert examples")):
        textlist = e.text_a
        subjects = e.labels

        #  logger.warning(f"textlist: {textlist}")
        tokens = tokenizer.tokenize(textlist)
        #  tokens = tokenizer.tokenize(''.join(textlist))[1:-1]

        #  logger.debug(f"tokens: {tokens}")
        #  logger.debug(f"textlist: {textlist}")
        #  logger.warning(
        #      f"tokens: {len(tokens)} - {tokens[:10]} - {tokens[-10:]}")
        #  logger.warning(
        #      f"textlist: {len(textlist)} - {textlist[:10]} - {textlist[-10:]}")
        #  assert len(tokens) == len(textlist)

        #  logger.debug(f"len(tokens): {len(tokens)}")
        start_ids = [0] * len(tokens)
        end_ids = [0] * len(tokens)
        subjects_id = []
        if subjects:
            for subject in subjects:
                #  logger.debug(f"subject: {subject}")
                label = subject[0]
                start = subject[1]
                end = subject[2]
                #  logger.warning(
                #      f"len(tokens): {len(tokens)}, subject: {subject}")
                start_ids[start] = label2id[label]
                end_ids[end] = label2id[label]
                subjects_id.append((label2id[label], start, end))
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            start_ids = start_ids[:(max_seq_length - special_tokens_count)]
            end_ids = end_ids[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        start_ids += [0]
        end_ids += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            tokens += [cls_token]
            start_ids += [0]
            end_ids += [0]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            start_ids = [0] + start_ids
            end_ids = [0] + end_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] *
                          padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] *
                           padding_length) + segment_ids
            start_ids = ([0] * padding_length) + start_ids
            end_ids = ([0] * padding_length) + end_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            start_ids += ([0] * padding_length)
            end_ids += ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length

        if ex_index < 3:
            logger.debug("*** Example ***")
            logger.debug(f"guid: {e.guid}")
            logger.debug(f"tokens: {' '.join([str(x) for x in tokens])}")
            logger.debug(f"input_ids: {' '.join([str(x) for x in input_ids])}")
            logger.debug(
                f"input_mask: {' '.join([str(x) for x in input_mask])}")
            logger.debug(
                f"segment_ids: {' '.join([str(x) for x in segment_ids])}")
            logger.info(f"start_ids: {' '.join([str(x) for x in start_ids])}")
            logger.info(f"end_ids: {' '.join([str(x) for x in end_ids])}")

            #  logger.info("*** Example ***")
            #  logger.info(f"guid: %s", e.guid)
            #  logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            #  logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            #  logger.info("input_mask: %s",
            #              " ".join([str(x) for x in input_mask]))
            #  logger.info("segment_ids: %s",
            #              " ".join([str(x) for x in segment_ids]))
            #  logger.info("start_ids: %s" % " ".join([str(x)
            #                                          for x in start_ids]))
            #  logger.info("end_ids: %s" % " ".join([str(x) for x in end_ids]))

        features.append(
            InputFeature(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         start_ids=start_ids,
                         end_ids=end_ids,
                         subjects=subjects_id,
                         input_len=input_len))
    return features


def features_to_dataset(features):
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features],
                                   dtype=torch.long)
    all_start_ids = torch.tensor([f.start_ids for f in features],
                                 dtype=torch.long)
    all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
    all_input_lens = torch.tensor([f.input_len for f in features],
                                  dtype=torch.long)
    #  all_subjects_ids = torch.tensor([f.subjects for f in features],
    #                                  dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_start_ids, all_end_ids, all_input_lens)
    return dataset


def examples_to_dataset_with_cache(args,
                                   examples,
                                   tokenizer,
                                   max_seq_length,
                                   evaluate=False,
                                   alias=""):

    # Make sure only the first process in distributed training process the dataset,
    # and the others will use the cache
    if not evaluate:
        barrier_member_processes(args)

    # Load data features from cache or dataset file
    cached_features_file = Path(
        args.data_dir
    ) / f"cached_ner_{args.dataset_name}{'_' + alias if alias else ''}"
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
            max_seq_length=max_seq_length,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            pad_on_left=bool(args.model_type in ['xlnet']),
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token
                                                       ])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.cache_features and is_master_process(args):
            logger.info(
                f"Saving features into cached file {cached_features_file}")
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset,
    # and the others will use the cache
    if not evaluate:
        barrier_leader_process(args)

    dataset = features_to_dataset(features)

    return dataset, features


def examples_to_dataset_old(examples, label2id, tokenizer, max_seq_length):

    features = convert_examples_to_features(
        examples,
        label2id,
        tokenizer,
        max_seq_length=max_seq_length,
        cls_token_at_end=False,
        pad_on_left=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)

    dataset = features_to_dataset(features)

    return dataset, features
