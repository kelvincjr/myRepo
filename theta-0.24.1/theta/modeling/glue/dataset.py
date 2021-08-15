#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm

from ...utils import seg_generator
from ...utils.multiprocesses import (barrier_leader_process,
                                     barrier_member_processes,
                                     is_master_process, is_multi_processes)
from ..glue_utils import InputExample
from ..trainer import common_batch_encode, common_to_tensors


class InputFeature(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """
    def __init__(self,
                 input_ids,
                 attention_mask=None,
                 token_type_ids=None,
                 input_len=None,
                 token_offsets=None,
                 label=None,
                 text=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.token_offsets = token_offsets
        self.label = label
        self.text = text

    def __repr__(self):
        return str(self.to_json_string())

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
#      for guid, text_a, text_b, label in data_generator(args,
#                                                        examples_file,
#                                                        seg_len=seg_len,
#                                                        seg_backoff=seg_backoff):
#          #  for (seg_text_a, seg_text_b) in seg_generator((text_a, text_b),
#          #                                                seg_len, seg_backoff):
#          #      #  seg_text = seg_text[0]
#          #      examples.append(
#          #          InputExample(guid=guid,
#          #                       text_a=seg_text_a,
#          #                       text_b=seg_text_b,
#          #                       label=label))
#          examples.append(
#              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#      logger.info(f"Loaded {len(examples)} examples from {examples_file}.")
#
#      return examples


def encode_examples(examples, label2id, tokenizer, max_seq_length):

    num_labels = len(label2id)
    texts = [e.text_a[:max_seq_length - 2] for e in examples]

    all_tokens, all_token2char, all_char2token, all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets = common_batch_encode(
        texts, label2id, tokenizer, max_seq_length)

    #  all_labels = [label2id[e.label] if e.label else 0 for e in examples]
    all_labels = []
    for e in examples:
        if isinstance(e.label, list):
            targets = [0] * len(label2id)
            for x in e.label:
                targets[label2id[x]] = 1
            all_labels.append(targets)
        else:
            if e.label:
                all_labels.append(label2id[e.label])
            else:
                all_labels.append(0)

    #  all_input_ids = torch.from_numpy(np.array(all_input_ids, dtype=np.int64))
    #  all_attention_mask = torch.from_numpy(
    #      np.array(all_attention_mask, dtype=np.int64))
    #  all_token_type_ids = torch.from_numpy(
    #      np.array(all_token_type_ids, dtype=np.int64))
    #  all_input_lens = torch.from_numpy(np.array(all_input_lens, dtype=np.int64))
    #  all_token_offsets = torch.from_numpy(
    #      np.array(all_token_offsets, dtype=np.int64))
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
                     label=label,
                     text=text) for input_ids, attention_mask,
        token_type_ids, input_len, token_offsets, label, text in zip(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_input_lens, all_token_offsets, all_labels, texts)
    ]

    return all_features
    #  all_features = [
    #      InputFeature(input_ids=input_ids,
    #                    attention_mask=attention_mask,
    #                    token_type_ids=token_type_ids,
    #                    label=label)
    #      for input_ids, attention_mask, token_type_ids, label in zip(
    #          all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    #  ]
    #  return (all_features, {
    #      'input_ids':
    #      torch.tensor(all_input_ids, dtype=torch.long),
    #      'attention_mask':
    #      torch.tensor(all_attention_mask, dtype=torch.long),
    #      'token_type_ids':
    #      torch.tensor(all_token_type_ids, dtype=torch.long),
    #      'lens':
    #      torch.tensor(all_input_lens, dtype=torch.long),
    #      'labels':
    #      torch.tensor(all_labels, dtype=torch.long),
    #  })


#  def encode_examples(examples, label2id, tokenizer, max_seq_length):
#
#      texts_a = [e.text_a for e in examples]
#      texts_b = [e.text_b for e in examples]
#      if not any(texts_b):
#          texts_b = None
#      #  texts_b = None
#      #  if len(examples) > 0 and examples[0].text_b:
#      #      texts_b = [e.text_b for e in examples]
#
#      if texts_b:
#          outputs = tokenizer.batch_encode_plus(texts_a,
#                                                text_pair=texts_b,
#                                                max_length=max_seq_length,
#                                                add_special_tokens=True,
#                                                return_tensors=None,
#                                                pad_to_max_length=True,
#                                                return_attention_mask=True,
#                                                return_token_type_ids=True)
#          all_input_ids, all_attention_mask, all_token_type_ids = outputs[
#              'input_ids'], outputs['attention_mask'], outputs['token_type_ids']
#          all_input_lens = [len(input_ids) for input_ids in all_input_ids]
#      else:
#          all_tokens = [
#              tokenizer.tokenize(text)[:max_seq_length]
#              for text in tqdm(texts_a, desc="Tokenize")
#          ]
#          all_input_ids = [
#              tokenizer.convert_tokens_to_ids(tokens) for tokens in all_tokens
#          ]
#          all_attention_mask = [[1] * len(input_ids)
#                                for input_ids in all_input_ids]
#          all_token_type_ids = [[0] * len(input_ids)
#                                for input_ids in all_input_ids]
#
#          all_input_lens = [len(input_ids) for input_ids in all_input_ids]
#
#          all_padding_lens = [max_seq_length - n for n in all_input_lens]
#          for i, (input_ids, attention_mask, token_type_ids,
#                  padding_length) in enumerate(
#                      zip(all_input_ids, all_attention_mask, all_token_type_ids,
#                          all_padding_lens)):
#              all_input_ids[i] = input_ids + [0] * padding_length
#              all_attention_mask[i] = attention_mask + [0] * padding_length
#              all_token_type_ids[i] = token_type_ids + [0] * padding_length
#
#
#      #  all_labels = [label2id[e.label] if e.label else 0 for e in examples]
#
#      all_labels = []
#      for e in examples:
#          if isinstance(e.label, list):
#              targets = [0] * len(label2id)
#              for x in e.label:
#                  targets[label2id[x]] = 1
#              all_labels.append(targets)
#          else:
#              if e.label:
#                  all_labels.append(label2id[e.label])
#              else:
#                  all_labels.append(0)
#
#      all_features = [
#          InputFeature(input_ids=input_ids,
#                        attention_mask=attention_mask,
#                        token_type_ids=token_type_ids,
#                        label=label)
#          for input_ids, attention_mask, token_type_ids, label in zip(
#              all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
#      ]
#      return (all_features, {
#          'input_ids':
#          torch.tensor(all_input_ids, dtype=torch.long),
#          'attention_mask':
#          torch.tensor(all_attention_mask, dtype=torch.long),
#          'token_type_ids':
#          torch.tensor(all_token_type_ids, dtype=torch.long),
#          'lens':
#          torch.tensor(all_input_lens, dtype=torch.long),
#          'labels':
#          torch.tensor(all_labels, dtype=torch.long),
#      })


def examples_to_dataset(examples, label2id, tokenizer, max_seq_length):

    all_features, outputs = encode_examples(examples, label2id, tokenizer,
                                            max_seq_length)

    dataset = TensorDataset(
        outputs['input_ids'],
        outputs['attention_mask'],
        outputs['token_type_ids'],
        outputs['lens'],
        outputs['labels'],
    )

    return dataset, all_features


#  def convert_examples_to_features(
#      examples,
#      label2id,
#      tokenizer,
#      max_length,
#      output_mode="classification",
#      pad_on_left=False,
#      pad_token=0,
#      pad_token_segment_id=0,
#      mask_padding_with_zero=True,
#  ):
#      """
#      Loads a data file into a list of ``InputFeature``
#
#      Args:
#          examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
#          tokenizer: Instance of a tokenizer that will tokenize the examples
#          max_length: Maximum example length
#          task: GLUE task
#          glue_labels: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
#          output_mode: String indicating the output mode. Either ``regression`` or ``classification``
#          pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
#          pad_token: Padding token
#          pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
#          mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
#              and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
#              actual values)
#
#      Returns:
#          If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
#          containing the task-specific features. If the input is a list of ``InputExamples``, will return
#          a list of task-specific ``InputFeature`` which can be fed to the model.
#
#      """
#
#      features = []
#      for (ex_index, example) in enumerate(tqdm(examples)):
#          len_examples = len(examples)
#          #  if ex_index % 10000 == 0:
#          #      logger.info("Writing example %d/%d" % (ex_index, len_examples))
#
#          inputs = tokenizer.encode_plus(
#              example.text_a,
#              example.text_b,
#              add_special_tokens=True,
#              max_length=max_length,
#              return_token_type_ids=True,
#          )
#          input_ids, token_type_ids = inputs["input_ids"], inputs[
#              "token_type_ids"]
#
#          # The mask has 1 for real tokens and 0 for padding tokens. Only real
#          # tokens are attended to.
#          attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
#
#          # Zero-pad up to the sequence length.
#          padding_length = max_length - len(input_ids)
#          if pad_on_left:
#              input_ids = ([pad_token] * padding_length) + input_ids
#              attention_mask = ([0 if mask_padding_with_zero else 1] *
#                                padding_length) + attention_mask
#              token_type_ids = ([pad_token_segment_id] *
#                                padding_length) + token_type_ids
#          else:
#              input_ids = input_ids + ([pad_token] * padding_length)
#              attention_mask = attention_mask + (
#                  [0 if mask_padding_with_zero else 1] * padding_length)
#              token_type_ids = token_type_ids + ([pad_token_segment_id] *
#                                                 padding_length)
#
#          assert len(input_ids
#                     ) == max_length, "Error with input length {} vs {}".format(
#                         len(input_ids), max_length)
#          assert len(attention_mask
#                     ) == max_length, "Error with input length {} vs {}".format(
#                         len(attention_mask), max_length)
#          assert len(token_type_ids
#                     ) == max_length, "Error with input length {} vs {}".format(
#                         len(token_type_ids), max_length)
#
#          label = 0
#          if example.label is not None:
#              if output_mode == "classification":
#                  label = label2id[example.label]
#              elif output_mode == "regression":
#                  label = float(example.label)
#              else:
#                  raise KeyError(output_mode)
#
#          if ex_index < 3:
#              logger.info("*** Example ***")
#              logger.info("guid: %s" % (example.guid))
#
#              tokens_a = tokenizer.tokenize(example.text_a)
#              logger.info(f"tokens_a: {' '.join([str(x) for x in tokens_a])}")
#              if example.text_b:
#                  tokens_b = tokenizer.tokenize(example.text_b)
#                  logger.info(
#                      f"tokens_b: {' '.join([str(x) for x in tokens_b])}")
#
#              logger.info("input_ids: %s" % " ".join([str(x)
#                                                      for x in input_ids]))
#              logger.info("attention_mask: %s" %
#                          " ".join([str(x) for x in attention_mask]))
#              logger.info("token_type_ids: %s" %
#                          " ".join([str(x) for x in token_type_ids]))
#              logger.info("label: %s (id = %d)" % (example.label, label))
#
#          features.append(
#              InputFeature(input_ids=input_ids,
#                            attention_mask=attention_mask,
#                            token_type_ids=token_type_ids,
#                            label=label))
#
#      logger.info(f"Leave {len(features)} features")
#      return features
#
#
#  def features_to_dataset(features):
#      # Convert to Tensors and build dataset
#      all_input_ids = torch.tensor([f.input_ids for f in features],
#                                   dtype=torch.long)
#      all_attention_mask = torch.tensor([f.attention_mask for f in features],
#                                        dtype=torch.long)
#      all_token_type_ids = torch.tensor([f.token_type_ids for f in features],
#                                        dtype=torch.long)
#      #  if output_mode == "regression":
#      all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
#      #  elif output_mode == "regression":
#      #      all_labels = torch.tensor([f.label for f in features],
#      #                                dtype=torch.float)
#
#      dataset = TensorDataset(all_input_ids, all_attention_mask,
#                              all_token_type_ids, all_labels)
#      return dataset
#

#  def examples_to_dataset_old(args,
#                              examples,
#                              tokenizer,
#                              max_seq_length,
#                              evaluate=False,
#                              output_mode="classification",
#                              alias=""):
#      # Make sure only the first process in distributed training process the dataset,
#      # and the others will use the cache
#      if not evaluate:
#          barrier_member_processes(args)
#
#      # Load data features from cache or dataset file
#      if args.data_dir:
#          cached_features_file = Path(
#              args.data_dir
#          ) / f"cached_glue_{args.dataset_name}{'_' + alias if alias else ''}"
#      else:
#          cached_features_file = None
#
#      if args.cache_features and cached_features_file and cached_features_file.exists(
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
#              max_length=max_seq_length,
#              output_mode=output_mode,
#              pad_on_left=bool(
#                  args.model_type in ["xlnet"]),  # pad on the left for xlnet
#              pad_token=tokenizer.pad_token_id,
#              pad_token_segment_id=tokenizer.pad_token_type_id,
#          )
#
#          if args.cache_features and cached_features_file and is_master_process(
#                  args):
#              logger.info(
#                  f"Saving features into cached file {cached_features_file}")
#              torch.save(features, cached_features_file)
#
#      logger.info(
#          f"convert {len(examples)} examples to {len(features)} features.")
#      # Make sure only the first process in distributed training process the dataset,
#      # and the others will use the cache
#      if not evaluate:
#          barrier_leader_process(args)
#
#      dataset = features_to_dataset(features)
#
#      return dataset

#  # -------------------- GlueExamples --------------------
#  class GlueExamples:
#      def __init__(self, filename, glue_labels, name):
#          self.filename = filename
#          self.name = name
#          self.glue_labels = glue_labels
#          self.examples = self._create_examples(self._read_txt(filename))
#
#          self._idx = 0
#
#      def __iter__(self):
#          return self
#
#      def __next__(self):
#          if self._idx < len(self.examples):
#              e = self.examples[self._idx]
#              self._idx += 1
#              return e
#          else:
#              raise StopIteration
#
#      def __get_item__(self, idx):
#          return self.examples[idx]
#
#      def __len__(self):
#          return len(self.examples)
#
#
# -------------------- DualSentencesDataset --------------------

#  class DualSentencesDataset:
#      """
#      原生只读训练数据集，用于生成模型训练集、验证集、测试集。
#      缺省文件名train_dataset.csv，必须有header，分隔符缺省为'\t'，
#      字段['sid', text_a', 'text_b', 'label', 'category']。
#      """
#
#      columns = ['sid', 'text_a', 'text_b', 'label', 'category']
#
#      def __init__(self,
#                   samples: [list, np.ndarray] = None,
#                   name: str = "default"):
#          self.name = name
#          self.columns = ['sid', 'text_a', 'text_b', 'label', 'category']
#          self.labels = []
#          self.categories = []
#          self.samples = pd.DataFrame([], columns=self.columns)
#
#          if samples is not None:
#              self.read_data(samples)
#
#      def __len__(self):
#          return self.samples.shape[0]
#
#      def info(self):
#          num_samples = self.samples.shape[0]
#
#          t1 = [len(x.text_a) for x in self.samples.itertuples()]
#          t1_mean = np.mean(t1)
#          t1_std = np.std(t1)
#          t1_min = np.min(t1)
#          t1_max = np.max(t1)
#          t2 = [len(x.text_b) for x in self.samples.itertuples()]
#          t2_mean = np.mean(t2)
#          t2_std = np.std(t2)
#          t2_min = np.min(t2)
#          t2_max = np.max(t2)
#
#          print(f"======== {self.name} ========")
#          print(f"Total {num_samples} samples.")
#          if num_samples == 0:
#              return
#
#          print(f"-------- sid --------")
#          num_sid = len(list(set(self.samples.sid.unique())))
#          print(f"{num_sid} unique sid.")
#          num_nan_sid = np.sum(
#              [len(x.sid) == 0 for x in self.samples.itertuples()])
#          print(f"{num_nan_sid} NaN sid.")
#
#          print(f"-------- text_a --------")
#          print(
#              f"mean: {t1_mean:.2f} std: {t1_std:.2f} min: {t1_min} max: {t1_max}"
#          )
#          print(f"-------- text_b --------")
#          print(
#              f"mean: {t1_mean:.2f} std: {t1_std:.2f} min: {t1_min} max: {t1_max}"
#          )
#          print(f"-------- labels --------")
#          print(f"{self.labels}")
#          if len(self.labels) > 0:
#              for label in self.labels:
#                  label_samples = np.sum(self.samples.label == label)
#                  print(
#                      f"{label}: {label_samples} ({label_samples/num_samples:.2f})"
#                  )
#
#          print(f"-------- categories --------")
#          print(f"{self.categories}")
#          if len(self.categories) > 0:
#              for category in self.categories:
#                  category_samples = np.sum(self.samples.category == category)
#                  print(
#                      f"{category}: {category_samples} ({category_samples/num_samples:.2f})"
#                  )
#
#      def read_csv(self,
#                   csv_file: str,
#                   default_label='-1',
#                   default_category='-1',
#                   sep='\t'):
#          df_samples = pd.read_csv(csv_file, sep=sep, dtype=np.str)
#          if 'label' not in df_samples.columns:
#              df_samples['label'] = [default_label] * df_samples.shape[0]
#          if 'category' not in df_samples.columns:
#              df_samples['category'] = [default_category] * df_samples.shape[0]
#
#          if not self._check_has_all_needed_columns(df_samples):
#              raise TypeError(
#                  f"Columns of samples must contain all columns in {self.columns}"
#              )
#
#          self.samples = df_samples.astype(np.str)
#          self._init_samples()
#
#          return True
#
#      def read_data(self, samples: [list, np.ndarray]):
#          if isinstance(samples, list):
#              samples = np.array(samples)
#          print(samples.shape)
#
#          if samples.shape[1] != len(self.columns):
#              raise TypeError(
#                  f"Columns of samples must contain all columns in {self.columns}"
#              )
#          self.samples = pd.DataFrame(samples, columns=self.columns)
#
#          self._init_samples()
#
#          return True
#
#      def _init_samples(self):
#          self.samples.text_a.fillna("")
#          self.samples.text_b.fillna("")
#          self.samples.label.fillna("")
#          self.samples.category.fillna("")
#
#          self.samples.info()
#
#          self.labels = sorted(list(set(self.samples.label.unique())))
#          self.categories = sorted(list(set(self.samples.category.unique())))
#
#      def to_csv(self,
#                 csv_file: str,
#                 columns: list = [],
#                 sep='\t',
#                 shuffle=False,
#                 random_state=None):
#
#          if len(columns) > 0:
#              samples = self.samples[columns]
#          else:
#              samples = self.samples
#              columns = self.columns
#          samples = samples.to_numpy()
#
#          if shuffle:
#              samples = shuffle_list(samples, random_state=random_state)
#
#          df_samples = pd.DataFrame(samples, columns=columns)
#          df_samples[['sid', 'text_a', 'text_b', 'label',
#                      'category']].to_csv(csv_file, index=False, sep=sep)
#          print(f"Dataset {self.name} saved to {csv_file}.")
#
#      def _check_has_all_needed_columns(self, df_samples: pd.DataFrame):
#          return np.sum([x in df_samples.columns
#                         for x in self.columns]) == len(self.columns)
#
#      def split_label(self,
#                      label: str,
#                      rates: list = [0.5],
#                      random_state=None) -> list:
#          samples = self.samples[self.samples.label == label].to_numpy()
#          samples = shuffle_list(samples, random_state=random_state)
#
#          nos = [int(samples.shape[0] * r) for r in rates]
#          last = samples.shape[0] - np.sum(nos)
#          assert (last > 0)
#          nos += [last]
#          pos = np.add.accumulate(nos)
#
#          samples_list = []
#          p0 = 0
#          for p1 in pos:
#              samples_list.append(samples[p0:p1])
#              p0 = p1
#
#          return samples_list
#
#      def split(self, rates: list = [0.5], random_state=None) -> list:
#          num_rates = len(rates) + 1
#          label_samples = []
#          for label in self.labels:
#              X = self.split_label(label, rates, random_state)
#              label_samples.append(X)
#
#          rate_samples = []
#          for i in range(num_rates):
#              X = []
#              for j, label in enumerate(self.labels):
#                  for row in label_samples[j][i]:
#                      X.append(row)
#              rate_samples.append(X)
#
#          datasets = []
#          for i in range(num_rates):
#              rate_samples[i] = shuffle_list(rate_samples[i],
#                                             random_state=random_state)
#              datasets.append(DualSentencesDataset(rate_samples[i]))
#
#          return datasets
#
#          #  samples = self.samples.to_numpy()
#          #  samples = shuffle_list(samples, random_state=random_state)
#          #
#          #  nos = [int(samples.shape[0] * r) for r in rates]
#          #  last = samples.shape[0] - np.sum(nos)
#          #  if last > 0:
#          #      nos += [last]
#          #  pos = np.add.accumulate(nos)
#          #
#          #  datasets = []
#          #  p0 = 0
#          #  for p1 in pos:
#          #      datasets.append(DualSentencesDataset(samples[p0:p1]))
#          #      p0 = p1
#          #
#          #  return datasets
#
#
#  #  class DualSentencesProcessor(DataProcessor):
#  #      def __init__(self, train_rate=0.9, train_fold=0):
#  #          self.ds_alldata = None
#  #          self.ds_test = None
#  #          self.train_rate = train_rate
#  #          self.fold = train_fold
#  #
#  #      def _load_train_dataset(self, data_dir):
#  #          self.ds_alldata = read_dataset(
#  #              os.path.join(data_dir, "train_dataset.csv"))
#  #          self.ds_train, self.ds_dev = self.ds_alldata.split(
#  #              rates=[self.train_rate], random_state=8864)
#  #
#  #          if self.fold > 0:
#  #              train_samples = self.ds_train.samples.to_numpy().tolist()
#  #              dev_samples = self.ds_dev.samples.to_numpy().tolist()
#  #              n = len(dev_samples)
#  #              p0 = n * (self.fold - 1)
#  #              p1 = p0 + n
#  #
#  #              #  tmp = train_samples[p0:p1]
#  #              #  train_samples[p0:p1] = dev_samples
#  #              #  dev_samples = tmp
#  #
#  #              train_samples[p0:p1], dev_samples = dev_samples, train_samples[
#  #                  p0:p1]
#  #
#  #              self.ds_train = DualSentencesDataset(train_samples)
#  #              self.ds_dev = DualSentencesDataset(dev_samples)
#  #
#  #      def _load_test_dataset(self, data_dir):
#  #          test_file = "test_dataset.csv"
#  #
#  #          test_data_file = os.path.join(data_dir, test_file)
#  #          print(f"test file: {test_file}")
#  #          test_data = pd.read_csv(test_data_file, sep='\t')
#  #          test_data.columns = ['sid', 'text_a', 'text_b', 'label', 'category']
#  #          test_data['label'] = ['0'] * test_data.shape[0]
#  #          test_data['category'] = ['0'] * test_data.shape[0]
#  #          self.ds_test = DualSentencesDataset(test_data)
#  #          #  self.ds_test = DualSentencesDataset()
#  #          #  self.ds_test.read_csv(test_data_file, default_label='0')
#  #
#  #      def get_example_from_tensor_dict(self, tensor_dict):
#  #          """See base class."""
#  #          return InputExample(tensor_dict['sid'].numpy(),
#  #                              tensor_dict['text_a'].numpy().decode('utf-8'),
#  #                              tensor_dict['text_b'].numpy().decode('utf-8'),
#  #                              str(tensor_dict['label'].numpy()))
#  #
#  #      def get_train_examples(self, data_dir):
#  #          if self.ds_alldata is None:
#  #              self._load_train_dataset(data_dir)
#  #          return self._create_examples(self.ds_train)
#  #
#  #      def get_dev_examples(self, data_dir):
#  #          if self.ds_alldata is None:
#  #              self._load_train_dataset(data_dir)
#  #          return self._create_examples(self.ds_dev)
#  #
#  #      def get_test_examples(self, data_dir):
#  #          if self.ds_test is None:
#  #              self._load_test_dataset(data_dir)
#  #          return self._create_examples(self.ds_test)
#  #
#  #      def get_labels(self):
#  #          return ["0", "1"]
#  #
#  #      def _create_examples(self, ds_examples):
#  #          examples = []
#  #          for (i, line) in enumerate(
#  #                  ds_examples.samples[['sid', 'text_a', 'text_b',
#  #                                       'label']].to_numpy()):
#  #              guid = line[0]
#  #              text_a = line[1]
#  #              text_b = line[2]
#  #              label = line[-1]
#  #              examples.append(
#  #                  InputExample(guid=guid,
#  #                               text_a=text_a,
#  #                               text_b=text_b,
#  #                               label=label))
#  #          return examples
