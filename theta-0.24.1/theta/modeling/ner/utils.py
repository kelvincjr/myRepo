#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, json
from loguru import logger
from collections import Counter
import torch
from transformers import BertTokenizer


class CNerTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=False, is_english=False):
        super().__init__(vocab_file=str(vocab_file),
                         do_lower_case=do_lower_case)
        self.vocab_file = str(vocab_file)
        self.do_lower_case = do_lower_case
        self.is_english = is_english

    def tokenize(self, text):
        if self.is_english:
            text_tokens = []
            words = text.split(' ')
            for w in words:
                word_tokens = super(CNerTokenizer, self).tokenize(w)
            text_tokens.extend(word_tokens)
        else:
            text_tokens = [c for c in text]

        _tokens = []

        for c in text_tokens:
            if self.do_lower_case:
                c = c.lower()
            if c in self.vocab:
                _tokens.append(c)
            else:
                _tokens.append('[UNK]')

        return _tokens


class DataProcessor(object):
    def __init__(self):
        self.lines = []

    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json list file."""
        with open(input_file, "r") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(json.loads(line.strip()))
            return lines

    @classmethod
    def _read_text(self, input_file):
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

        self.lines = lines
        return lines


#  def get_entity_bios_new(seq, id2label):
#      """Gets entities from sequence.
#      note: BIOS
#      Args:
#          seq (list): sequence of labels.
#      Returns:
#          list: list of (chunk_type, chunk_start, chunk_end).
#      Example:
#          # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
#          # >>> get_entity_bios(seq)
#          [['PER', 0,1], ['LOC', 3, 3]]
#      """
#      hanging = -1
#      chunks = []
#      chunk = [-1, -1, -1]
#      for indx, tag in enumerate(seq):
#          if not isinstance(tag, str):
#              tag = id2label[tag]
#          if tag.startswith("S-"):
#              if chunk[2] != -1:
#                  chunks.append(chunk)
#              chunk = [-1, -1, -1]
#              chunk[1] = indx
#              chunk[2] = indx
#              chunk[0] = ''.join(tag.split('-')[1:])
#              chunks.append(chunk)
#              chunk = [-1, -1, -1]
#              hanging = -1
#          if tag.startswith("B-"):
#              if chunk[2] != -1:
#                  chunks.append(chunk)
#              chunk = [-1, -1, -1]
#              chunk[1] = indx
#              chunk[0] = '-'.join(tag.split('-')[1:])
#              hangle = -1
#          elif tag.startswith('I-') and chunk[1] != -1:
#              _type = '-'.join(tag.split('-')[1:])
#              #  if _type == chunk[0]:
#              chunk[2] = indx
#              if indx == len(seq) - 1:
#                  chunks.append(chunk)
#              hanging = 0
#          elif tag.startswith('I-') and chunk[1] == -1 and indx > 0:
#              _type = '-'.join(tag.split('-')[1:])
#              chunk[0] = _type
#              chunk[1] = indx - 1
#              chunk[2] = indx
#              hanging = 0
#          else:
#              if hanging >= 0:
#                  hanging += 1
#                  if hanging >= 3:
#                      if chunk[2] != -1:
#                          chunks.append(chunk)
#                      chunk = [-1, -1, -1]
#                      hanging = -1
#
#          #  else:
#          #      if chunk[2] != -1:
#          #          chunks.append(chunk)
#          #      chunk = [-1, -1, -1]
#      if chunk[2] != -1:
#          chunks.append(chunk)
#      return chunks


def fix_seq_IOOI(seq, id2label):
    # fix ["I-xxx", 'O', "I-xxx"] to ['I-xxx', 'I-xxx', 'I-xxx']
    if len(seq) < 4:
        return seq

    for i in range(0, len(seq) - 4):
        tag0, tag1, tag2 = seq[i:i + 4]
        #  if not isinstance(tag0, str):
        #      tag0 = id2label[tag0]
        #      tag1 = id2label[tag1]
        #      tag2 = id2label[tag2]
        if tag1 == 'O' and tag0.startswith('I-') and tag1.startswith('I-'):
            seq[i + 1] = tag0
            seq[i + 2] = tag0
            i += 3
    return seq


def fix_seq_BOI(seq, id2label):
    # fix ["I-xxx", 'O', "I-xxx"] to ['I-xxx', 'I-xxx', 'I-xxx']
    if len(seq) < 3:
        return seq

    for i in range(0, len(seq) - 3):
        tag0, tag1, tag2 = seq[i:i + 3]
        #  if not isinstance(tag0, str):
        #      tag0 = id2label[tag0]
        #      tag1 = id2label[tag1]
        #      tag2 = id2label[tag2]
        if tag1 == 'O' and tag0.startswith('B-') and tag2.startswith('I-'):
            seq[i + 1] = 'I-' + tag0[2:]
            i += 2
    return seq


def fix_seq_IOI(seq, id2label):
    # fix ["I-xxx", 'O', "I-xxx"] to ['I-xxx', 'I-xxx', 'I-xxx']
    if len(seq) < 3:
        return seq

    for i in range(0, len(seq) - 3):
        tag0, tag1, tag2 = seq[i:i + 3]
        #  if not isinstance(tag0, str):
        #      tag0 = id2label[tag0]
        #      tag1 = id2label[tag1]
        #      tag2 = id2label[tag2]
        if tag1 == 'O' and tag0.startswith('I-') and tag2.startswith('I-'):
            seq[i + 1] = tag0
            i += 2
    return seq


def fix_seq_OI(seq, id2label):
    # fix ["I-xxx", 'O', "I-xxx"] to ['I-xxx', 'I-xxx', 'I-xxx']
    if len(seq) < 2:
        return seq

    for i in range(0, len(seq) - 2):
        tag0, tag1 = seq[i:i + 2]
        #  if not isinstance(tag0, str):
        #      tag0 = id2label[tag0]
        #      tag1 = id2label[tag1]
        if tag0 == 'O' and tag1.startswith('I-'):
            seq[i] = 'B-' + tag1[2:]
            i += 2
    return seq


def fix_seq_OII(seq, id2label):
    # fix ["I-xxx", 'O', "I-xxx"] to ['I-xxx', 'I-xxx', 'I-xxx']
    if len(seq) < 3:
        return seq

    for i in range(0, len(seq) - 3):
        tag0, tag1, tag2 = seq[i:i + 3]
        #  if not isinstance(tag0, str):
        #      tag0 = id2label[tag0]
        #      tag1 = id2label[tag1]
        #      tag2 = id2label[tag2]
        if tag0 == 'O' and tag1.startswith('I-') and tag2.startswith('I-'):
            seq[i] = 'B-' + tag1[2:]
            i += 3
    return seq


def fix_seq_XOI(seq, id2label, num_O=1):
    if len(seq) < num_O + 2:
        return seq

    #  logger.debug(f"{seq}")
    for i in range(0, len(seq) - num_O - 1):
        tags = [x for x in seq[i:i + num_O + 2]]
        #  if not isinstance(tags[0], str):
        #      tags = [id2label[x] for x in tags]
        tags = [x[:1] for x in tags]
        if tags[0] != 'O' and tags[0] != 'S' and tags[-1] == 'I':
            if tags[1:num_O + 1] == ['O'] * num_O:
                for j in range(1, num_O + 1):
                    seq[i + j] = 'I-' + seq[i][2:]
    return seq


def get_entity_bios(seq, id2label, autofix=False):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """

    #  logger.debug(f"{seq}")
    #  seq = [id2label[x] for x in seq]
    if autofix:
        seq = fix_seq_XOI(seq, id2label, num_O=3)
        seq = fix_seq_XOI(seq, id2label, num_O=2)
        seq = fix_seq_XOI(seq, id2label, num_O=1)
        #  #  seq = fix_seq_BOI(seq, id2label)
        #  #  seq = fix_seq_IOI(seq, id2label)
        #  #  seq = fix_seq_OI(seq, id2label)
        seq = fix_seq_OII(seq, id2label)

    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        #  if not isinstance(tag, str):
        #  tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = '-'.join(tag.split('-')[1:])
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = '-'.join(tag.split('-')[1:])
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = '-'.join(tag.split('-')[1:])
            #  if _type == chunk[0]:
            #      chunk[2] = indx
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, id2label, markup='bios', autofix=False):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label, autofix)


def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return S


class SeqEntityScore(object):
    def __init__(self,
                 id2label,
                 ignore_categories=None,
                 markup='bios',
                 autofix=False):
        self.id2label = id2label
        self.ignore_categories = ignore_categories
        self.markup = markup
        self.autofix = autofix
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision *
                                                 recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        #  origin_counter = Counter([x[0] for x in self.origins])
        #  found_counter = Counter([x[0] for x in self.founds])
        #  right_counter = Counter([x[0] for x in self.rights])
        origin_counter = Counter([f"{x[0]}" for x in self.origins])
        found_counter = Counter([f"{x[0]}" for x in self.founds])
        right_counter = Counter([f"{x[0]}" for x in self.rights])

        total_origin = 0
        total_found = 0
        total_right = 0
        for type_, count in origin_counter.items():
            category = type_
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {
                "acc": round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'right': right,
                'found': found,
                'origin': origin
            }
            if self.ignore_categories and category in self.ignore_categories:
                pass
            else:
                total_origin += origin
                total_found += found
                total_right += right
        if self.ignore_categories:
            origin = total_origin
            found = total_found
            right = total_right
        else:
            origin = len(self.origins)
            found = len(self.founds)
            right = len(self.rights)
        #  origin = len(self.origins)
        #  found = len(self.founds)
        #  right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {
            'acc': precision,
            'recall': recall,
            'f1': f1,
            'right': right,
            'found': found,
            'origin': origin
        }, class_info

    def update(self, label_paths, pred_paths):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        #  logger.debug(f"label_paths: {label_paths}")
        #  logger.debug(f"pred_paths: {pred_paths}")
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path, self.id2label,
                                          self.markup, self.autofix)
            pre_entities = get_entities(pre_path, self.id2label, self.markup,
                                        self.autofix)
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([
                pre_entity for pre_entity in pre_entities
                if pre_entity in label_entities
            ])
