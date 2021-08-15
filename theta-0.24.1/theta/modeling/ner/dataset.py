#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm

from ...utils.multiprocesses import (barrier_leader_process,
                                     barrier_member_processes,
                                     is_multi_processes)
from ..ner_utils import InputExample
from ..trainer import common_batch_encode, common_to_tensors


class InputFeature(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask, token_type_ids, input_len,
                 token_offsets, labels, text):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.token_offsets = token_offsets

        self.labels = labels
        self.text = text

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        #  return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        return json.dumps(self.to_dict(), ensure_ascii=False,
                          sort_keys=True) + "\n"


def to_BIOS(labels, num_tokens, char2token):
    bios_labels = ['O'] * num_tokens
    #  logger.debug(f"num_tokens: {num_tokens}, labels: {labels}")
    for c, s, e in labels:
        assert s <= e
        if s < 0 or e < 0:
            logger.warning(f"num_tokens: {num_tokens}, labels: {labels}")
            continue
        s = char2token[s] - 1
        e = char2token[e] - 1
        if s == e:
            bios_labels[s] = f"S-{c}"
        else:
            bios_labels[s] = f"B-{c}"
            for i in range(1, e + 1 - s):
                bios_labels[s + i] = f"I-{c}"

    return bios_labels


def encode_examples(examples, label2id, tokenizer, max_seq_length):

    num_labels = len(label2id)
    texts = [e.text_a[:max_seq_length - 2] for e in examples]

    all_tokens, all_token2char, all_char2token, all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets = common_batch_encode(
        texts, label2id, tokenizer, max_seq_length)

    def encode_labels(example, labels, input_len, char2token):
        #  logger.warning(f"text: {example.text_a}")
        if labels is not None:
            #  logger.debug(f"input_len: {input_len}, labels: {labels}")
            label_ids = [
                label2id[x] for x in to_BIOS(labels, input_len, char2token)
            ]
        else:
            label_ids = [label2id['O']] * input_len
        return label_ids

    all_labels = [[label2id['[CLS]']] +
                  encode_labels(e, e.labels, input_len - 2, char2token) +
                  [label2id['[SEP]']] + [0] * (max_seq_length - input_len)
                  for e, input_len, char2token in zip(examples, all_input_lens,
                                                      all_char2token)]
    #  logger.warning(f"all_labels[0]: {all_labels[0]}")

    logger.debug(f"all_input_ids.shape: {np.array(all_input_ids).shape}")
    logger.debug(
        f"all_attention_mask.shape: {np.array(all_attention_mask).shape}")
    logger.debug(
        f"all_token_type_ids.shape: {np.array(all_token_type_ids).shape}")
    logger.debug(f"all_labels.shape: {np.array(all_labels).shape}")
    logger.debug(f"all_input_lens.shape: {np.array(all_input_lens).shape}")
    assert np.array(all_input_ids).shape[1] == max_seq_length
    assert np.array(all_attention_mask).shape[1] == max_seq_length
    assert np.array(all_token_type_ids).shape[1] == max_seq_length

    all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets = common_to_tensors(
        all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens,
        all_token_offsets)

    all_labels = torch.from_numpy(np.array(all_labels, dtype=np.int64))

    all_features = [
        InputFeature(input_ids=input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids,
                     input_len=input_len,
                     token_offsets=token_offsets,
                     labels=labels,
                     text=text) for input_ids, attention_mask,
        token_type_ids, input_len, token_offsets, text, labels in zip(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_input_lens, all_token_offsets, texts, all_labels)
    ]

    return all_features


def encode_examples_old(examples, label2id, tokenizer, max_seq_length):

    texts = [e.text_a[:max_seq_length - 2] for e in examples]

    #  for e in tqdm(examples, desc="to BIOS"):
    #      e.labels = to_BIOS(e.labels, len(e.text_a))

    #  logger.debug(f"texts: {texts}")

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

    #  all_text_lens = [len(text) for text in texts]
    #  logger.debug(f"all_text_lens: {all_text_lens}")

    all_input_lens = [len(tokens) for tokens in all_tokens]
    #  logger.debug(f"all_input_lens: {all_input_lens}")

    all_input_ids = [
        tokenizer.convert_tokens_to_ids(tokens) for tokens in all_tokens
    ]
    all_attention_mask = [[1] * len(input_ids) for input_ids in all_input_ids]
    all_token_type_ids = [[0] * len(input_ids) for input_ids in all_input_ids]

    def encode_labels(labels, input_len):
        if labels is not None:
            #  logger.debug(f"input_len: {input_len}, labels: {labels}")
            label_ids = [label2id[x] for x in to_BIOS(labels, input_len)]
        else:
            label_ids = [label2id['O']] * input_len
        return label_ids

    all_labels = [[label2id[tokenizer.cls_token]] +
                  encode_labels(e.labels, input_len - 2) +
                  [label2id[tokenizer.sep_token]]
                  for e, input_len in zip(examples, all_input_lens)]

    all_padding_lens = [max_seq_length - n for n in all_input_lens]
    for i, (input_ids, attention_mask, token_type_ids, label_ids,
            padding_length) in enumerate(
                zip(all_input_ids, all_attention_mask, all_token_type_ids,
                    all_labels, all_padding_lens)):
        all_input_ids[i] = input_ids + [0] * padding_length
        all_attention_mask[i] = attention_mask + [0] * padding_length
        all_token_type_ids[i] = token_type_ids + [0] * padding_length
        all_labels[i] = label_ids + [0] * padding_length

    logger.debug(f"all_input_ids.shape: {np.array(all_input_ids).shape}")
    logger.debug(
        f"all_attention_mask.shape: {np.array(all_attention_mask).shape}")
    logger.debug(
        f"all_token_type_ids.shape: {np.array(all_token_type_ids).shape}")
    logger.debug(f"all_labels.shape: {np.array(all_labels).shape}")
    logger.debug(f"all_input_lens.shape: {np.array(all_input_lens).shape}")
    assert np.array(all_input_ids).shape[1] == max_seq_length
    assert np.array(all_attention_mask).shape[1] == max_seq_length
    assert np.array(all_token_type_ids).shape[1] == max_seq_length

    all_features = [
        InputFeature(input_ids=input_ids,
                     input_mask=attention_mask,
                     segment_ids=token_type_ids,
                     label_ids=label_ids,
                     input_len=input_len)
        for input_ids, attention_mask, token_type_ids, label_ids, input_len in
        zip(all_input_ids, all_attention_mask, all_token_type_ids, all_labels,
            all_input_lens)
    ]

    return (all_features, {
        'input_ids':
        torch.tensor(all_input_ids, dtype=torch.long),
        'attention_mask':
        torch.tensor(all_attention_mask, dtype=torch.long),
        'token_type_ids':
        torch.tensor(all_token_type_ids, dtype=torch.long),
        'labels':
        torch.tensor(all_labels, dtype=torch.long),
        'input_lens':
        torch.tensor(all_input_lens, dtype=torch.long),
    })


#  def examples_to_dataset(examples, label2id, tokenizer, max_seq_length):
#      all_features, outputs = encode_examples(examples, label2id, tokenizer,
#                                              max_seq_length)
#
#      dataset = TensorDataset(
#          outputs['input_ids'],
#          outputs['attention_mask'],
#          outputs['token_type_ids'],
#          outputs['labels'],
#          outputs['input_lens'],
#      )
#
#      return dataset, all_features

#  def convert_examples_to_features(
#      examples,
#      label2id,
#      tokenizer,
#      max_seq_length=512,
#      cls_token_at_end=False,
#      cls_token="[CLS]",
#      cls_token_segment_id=1,
#      sep_token="[SEP]",
#      pad_on_left=False,
#      pad_token=0,
#      pad_token_segment_id=0,
#      sequence_a_segment_id=0,
#      mask_padding_with_zero=True,
#  ):
#      """ Loads a data file into a list of `InputBatch`s
#          `cls_token_at_end` define the location of the CLS token:
#              - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
#              - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
#          `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
#      """
#      #  label2id = {label: i for i, label in enumerate(examples.label_set)}
#      features = []
#      for ex_index, e in enumerate(tqdm(examples, desc="Convert examples")):
#          tokens = tokenizer.tokenize(e.text_a)
#          if e.labels is not None:
#              label_ids = [label2id[x] for x in e.labels]
#          else:
#              label_ids = [label2id['O']] * len(e.text_a)
#
#          # Account for [CLS] and [SEP] with "- 2".
#          special_tokens_count = 2
#          if len(tokens) > max_seq_length - special_tokens_count:
#              tokens = tokens[:(max_seq_length - special_tokens_count)]
#              label_ids = label_ids[:(max_seq_length - special_tokens_count)]
#
#          # The convention in BERT is:
#          # (a) For sequence pairs:
#          #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#          #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
#          # (b) For single sequences:
#          #  tokens:   [CLS] the dog is hairy . [SEP]
#          #  type_ids:   0   0   0   0  0     0   0
#          #
#          # Where "type_ids" are used to indicate whether this is the first
#          # sequence or the second sequence. The embedding vectors for `type=0` and
#          # `type=1` were learned during pre-training and are added to the wordpiece
#          # embedding vector (and position vector). This is not *strictly* necessary
#          # since the [SEP] token unambiguously separates the sequences, but it makes
#          # it easier for the model to learn the concept of sequences.
#          #
#          # For classification tasks, the first vector (corresponding to [CLS]) is
#          # used as as the "sentence vector". Note that this only makes sense because
#          # the entire model is fine-tuned.
#          tokens += [sep_token]
#          label_ids += [label2id[sep_token]]
#          segment_ids = [sequence_a_segment_id] * len(tokens)
#
#          if cls_token_at_end:
#              tokens += [cls_token]
#              label_ids += [label2id[cls_token]]
#              segment_ids += [cls_token_segment_id]
#          else:
#              tokens = [cls_token] + tokens
#              label_ids = [label2id[cls_token]] + label_ids
#              segment_ids = [cls_token_segment_id] + segment_ids
#
#          input_ids = tokenizer.convert_tokens_to_ids(tokens)
#          # The mask has 1 for real tokens and 0 for padding tokens. Only real
#          # tokens are attended to.
#          input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
#          input_len = len(label_ids)
#          # Zero-pad up to the sequence length.
#          padding_length = max_seq_length - len(input_ids)
#          if pad_on_left:
#              input_ids = ([pad_token] * padding_length) + input_ids
#              input_mask = ([0 if mask_padding_with_zero else 1] *
#                            padding_length) + input_mask
#              segment_ids = ([pad_token_segment_id] *
#                             padding_length) + segment_ids
#              label_ids = ([pad_token] * padding_length) + label_ids
#          else:
#              input_ids += [pad_token] * padding_length
#              input_mask += [0 if mask_padding_with_zero else 1] * padding_length
#              segment_ids += [pad_token_segment_id] * padding_length
#              label_ids += [pad_token] * padding_length
#
#          assert len(input_ids) == max_seq_length
#          assert len(input_mask) == max_seq_length
#          assert len(segment_ids) == max_seq_length
#          assert len(label_ids) == max_seq_length
#          if ex_index < 3:
#              logger.debug("*** Example ***")
#              logger.debug(f"guid: {e.guid}")
#              logger.debug(f"tokens: {' '.join([str(x) for x in tokens])}")
#              logger.debug(f"input_ids: {' '.join([str(x) for x in input_ids])}")
#              logger.debug(
#                  f"input_mask: {' '.join([str(x) for x in input_mask])}")
#              logger.debug(
#                  f"segment_ids: {' '.join([str(x) for x in segment_ids])}")
#              logger.debug(f"label_ids: {' '.join([str(x) for x in label_ids])}")
#
#          features.append(
#              InputFeature(input_ids=input_ids,
#                            input_mask=input_mask,
#                            input_len=input_len,
#                            segment_ids=segment_ids,
#                            label_ids=label_ids))
#      return features
#
#
#  def features_to_dataset(features):
#      # Convert to Tensors and build dataset
#      all_input_ids = torch.tensor([f.input_ids for f in features],
#                                   dtype=torch.long)
#      all_input_mask = torch.tensor([f.input_mask for f in features],
#                                    dtype=torch.long)
#      all_segment_ids = torch.tensor([f.segment_ids for f in features],
#                                     dtype=torch.long)
#      all_label_ids = torch.tensor([f.label_ids for f in features],
#                                   dtype=torch.long)
#      all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
#      dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
#                              all_lens, all_label_ids)
#      return dataset
#
#
#  def examples_to_dataset_old(args,
#                              examples,
#                              ner_labels,
#                              tokenizer,
#                              max_seq_length,
#                              evaluate=False,
#                              alias=""):
#      # Make sure only the first process in distributed training process the dataset,
#      # and the others will use the cache
#      if not evaluate:
#          barrier_member_processes(args)
#
#      label_set = ['X', 'O', '[CLS]', '[SEP]']
#      for x in ner_labels:
#          label_set += [f"B-{x}", f"I-{x}", f"S-{x}"]
#
#      args.id2label = {i: label for i, label in enumerate(label_set)}
#      args.label2id = {label: i for i, label in enumerate(label_set)}
#      args.num_labels = len(label_set)
#
#      # Load data features from cache or dataset file
#      cached_features_file = Path(
#          args.data_dir
#      ) / f"cached_ner_{args.dataset_name}{'_' + alias if alias else ''}"
#      if args.cache_features and cached_features_file.exists(
#      ) and not args.overwrite_cache:
#          logger.info(
#              f"Loading features from cached file {cached_features_file}")
#          features = torch.load(cached_features_file)
#      else:
#          logger.info(f"Creating features from dataset file at {args.data_dir}")
#
#          features = convert_examples_to_features(
#              examples,
#              args.label2id,
#              tokenizer,
#              max_seq_length=max_seq_length,
#              cls_token_at_end=bool(args.model_type in ["xlnet"]),
#              pad_on_left=bool(args.model_type in ['xlnet']),
#              cls_token=tokenizer.cls_token,
#              cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
#              sep_token=tokenizer.sep_token,
#              # pad on the left for xlnet
#              pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token
#                                                         ])[0],
#              pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
#          )
#
#          if args.cache_features and is_master_process(args):
#              logger.info(
#                  f"Saving features into cached file {cached_features_file}")
#              torch.save(features, cached_features_file)
#
#      # Make sure only the first process in distributed training process the dataset,
#      # and the others will use the cache
#      if not evaluate:
#          barrier_leader_process(args)
#
#      dataset = features_to_dataset(features)
#
#      return dataset


def load_examples_from_bios_file(filename):
    def _read_txt(input_file):
        lines = []
        with open(input_file, 'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    def _create_examples(lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #  if i == 0:
            #      continue
            guid = "%s-%s" % ("example", i)
            text_a = line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(
                InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

    return _create_examples(_read_txt(filename))


def export_bios_file(examples, bios_file):
    fw_bios = open(bios_file, 'w')
    for example in tqdm(examples):
        # for test examples
        labels = example.labels
        if labels is None:
            labels = ['O'] * len(example.text_a)

        for w, t in zip(example.text_a, labels):
            fw_bios.write(f"{w} {t}\n")
        fw_bios.write("\n")
    logger.info(f"export {len(examples)} examples to {bios_file}")


# ------ Generate CRF train data -------

[
    {
        'id': "",
        'text': "",
        'entities': [
            {
                'label': "",
                'start_pos': "",
                'end_pos': "",
            },
        ]
    },
]


class TaggedCRFData:
    def __init__(self, all_text_entities):
        self.all_text_entities = all_text_entities
        self.labels_map = []

    def generate_tagged_crf_data(self, num_augements=10, shuffle=False):
        lines = self._generate_crf_data()
        aug_lines = self._augement(num_augements, shuffle=shuffle)
        return lines + aug_lines

    def generate_tagged_text(self, text, entities):
        num_tokens = len(text)
        tokens = ['O'] * num_tokens
        p0 = 0
        for entity in entities:
            label_type = entity['label_type']
            s = entity['start_pos'] - 1
            e = entity['end_pos'] - 1
            print(s, e)
            assert e >= s
            if s >= len(text) or e >= len(text):
                continue

            entity_text = text[s:e + 1]
            print(label_type, entity_text)

            assert len(entity_text) > 0
            if label_type not in self.labels_map:
                self.labels_map[label_type] = []
            self.labels_map[label_type].append(entity_text)

            n = e - s + 1
            t = label2type[label_type]
            if n == 1:
                tokens[s] = f"S-{t}"
            else:
                tokens[s] = f"B-{t}"
                for i in range(1, n):
                    tokens[s + i] = f"I-{t}"
        return tokens

    def _generate(self):
        self.labels_map = {}
        lines = []
        for json_file, text, entities in self.all_text_entities:
            tokens = generate_tagged_text(text, entities, self.labels_map)
            chars = [c for c in text]
            for n, (c, tok) in enumerate(zip(chars, tokens)):
                #  Fout.write(f"{n+1:03d}: {c} {tok}\n")
                lines.append(f"{c} {tok}\n")
            lines.append("\n")
        return lines

    def _augement(self, num_augements=10, shuffle=False):
        aug_tokens = []
        for i, (json_file, text, entities) in enumerate(
                tqdm(self.all_text_entities,
                     desc=f"Augement {num_augements}X")):

            #  print(f"-------------------{json_file}--------------------")
            #  print(text)
            #  print(entities)
            #  for entity in entities:
            #      s = entity['start_pos'] - 1
            #      e = entity['end_pos']
            #      print(f"{entity['label_type']}: {text[s:e]}")
            #  print("----------------------------------------")
            for _ in range(num_augements):
                e_idx = random.randint(0, len(entities) - 1)
                entity = entities[e_idx]

                label_type = entity['label_type']
                s = entity['start_pos'] - 1
                e = entity['end_pos'] - 1

                labels = self.labels_map[label_type]
                idx = random.randint(0, len(labels) - 1)
                new_entity_text = labels[idx]

                text = text[:s] + new_entity_text + text[e:]
                #  print(entity, new_entity_text)
                assert len(new_entity_text) >= 0
                delta = len(new_entity_text) - (e - s + 1) + 1

                #  print(
                #      f"delta={delta} e_idx: {e_idx} new_entity_text: {new_entity_text}"
                #  )
                #  print(entities)
                entity['end_pos'] = entity['start_pos'] + len(
                    new_entity_text) - 1
                for n, e in enumerate(entities):
                    if n > e_idx:
                        e['start_pos'] += delta
                        e['end_pos'] += delta
                        assert e['start_pos'] <= e['end_pos']
                    else:
                        assert e['start_pos'] <= e['end_pos']
                #  print(f"-- Adjusted {entities}")
                #  print(text[:s] + "【" + new_entity_text + "】" +
                #        text[entity['end_pos'] + 1:])

                tokens = generate_tagged_text(text, entities)
                aug_tokens.append((text, tokens))

        if shuffle:
            random.shuffle(aug_tokens)

        lines = []

        for i, (text,
                tokens) in enumerate(tqdm(aug_tokens, desc='Write augements')):
            chars = [c for c in text]
            for n, (c, tok) in enumerate(zip(chars, tokens)):
                lines.append(f"{c} {tok}\n")
            lines.append("\n")

        return lines
