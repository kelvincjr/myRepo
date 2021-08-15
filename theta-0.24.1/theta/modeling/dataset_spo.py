#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPO数据集持久化，与标注文件之间的互转。

import json
import os
import re
from dataclasses import dataclass, field
from typing import List

from loguru import logger
from tqdm import tqdm


# 实体标签
@dataclass
class EntityTag:
    # 标签类别
    category: str = ""
    # 标签起始位置
    start: int = -1
    # 标签文本
    mention: str = ""

    def clear(self):
        pass

    def to_dict(self):
        return {
            'category': self.category,
            'start': self.start,
            'mention': self.mention
        }

    def from_dict(self, dict_data):
        self.clear()
        for k in self.__dict__.keys():
            if k in dict_data:
                v = dict_data[k]
                setattr(self, k, v)
        return self


# 关系标签
@dataclass
class RelTag:
    predicate: str = None
    sub: EntityTag = None
    obj: EntityTag = None

    def clear(self):
        self.predicate = None
        self.sub = None
        self.obj = None

    def to_dict(self):
        return {
            'predicate': self.predicate,
            'sub': self.sub.to_dict(),
            'obj': self.obj.to_dict()
        }

    def from_dict(self, dict_data):
        self.clear()
        self.predicate = dict_data.get('predicate', None)
        self.sub = dict_data.get('sub', None)
        if self.sub:
            self.sub = EntityTag().from_dict(self.sub)
        self.obj = dict_data.get('obj', None)
        if self.obj:
            self.obj = EntityTag().from_dict(self.obj)
        return self


# 打过标签的文本
@dataclass
class TaggedText:
    guid: str
    text: str
    spo_list: List[RelTag] = field(default_factory=list)

    def clear(self):
        self.spo_list = []

    def to_dict(self):
        return {
            'guid': self.guid,
            'text': self.text,
            'spo_list': [x.to_dict() for x in self.spo_list]
        }

    def from_dict(self, dict_data):
        self.clear()
        if 'guid' in dict_data:
            self.guid = dict_data['guid']
        elif 'text' in dict_data:
            self.text = dict_data['text']
        elif 'tags' in dict_data:
            for x in dict_data['spo_list']:
                self.spo_list.append(RelTag().from_dict(x))

        return self

    def add_tag(self, tag: RelTag):
        self.spo_list.append(tag)


# 实体关系标签数据集
class SpoDataset:
    def __init__(self, name=None, predicate_labels=None):
        self.name = name
        self.predicate_labels = predicate_labels
        self.tagged_text_list: List[TaggedText] = []

    def __len__(self):
        return len(self.tagged_text_list)

    def __getitem__(self, idx):
        return self.tagged_text_list[idx]

    def __iter__(self):
        for x in self.tagged_text_list:
            yield x

    def info(self):
        logger.info(f"Total: {len(self.tagged_text_list)}")

    def append(self, tagged_text: TaggedText):
        self.tagged_text_list.append(tagged_text)

    def extend(self, other_dataset):
        self.tagged_text_list.extend(other_dataset)

    def save(self, filename: str, format=None):
        """
        {'guid': '0000', 'text': "sample text", 'spo_list': [{'category': "实体类别1", start: 10, mention: "实体文本"}, ...]}
        """

        if format == "json":
            json_data = {x.guid: x.to_dict() for x in self.tagged_text_list}
            json.dump(json_data,
                      open(filename, 'w'),
                      ensure_ascii=False,
                      indent=2)
        else:
            with open(filename, 'w') as wt:
                for tagged_text in tqdm(self.tagged_text_list,
                                        desc="Save tagged text list."):
                    data_dict = tagged_text.to_dict()
                    wt.write(f"{json.dumps(data_dict, ensure_ascii=False)}\n")
        logger.info(f"Saved {filename}")

    def load_from_file(self, filename: str):
        logger.info(f"Loading {filename}")
        try:
            lines = [
                json.loads(x.strip()) for x in open(filename, 'r').readlines()
            ]
        except:
            logger.warning(
                f"{filename} is not standard style, try reviews style...")
            try:
                json_data = json.load(open(filename, 'r'))
                lines = [v for k, v in json_data.items()]
            except:
                logger.error("Unknown file style {filename}")
                raise Exception(f"Unknown file style {filename}")

        def data_generator(data_source=None):
            for json_data in lines:
                yield json_data['guid'], json_data['text'], None, json_data[
                    'tags']

        return self.load(data_generator)

    def load_from_brat_data(self, brat_data_dir):
        from .brat import brat_data_generator
        return self.load(brat_data_generator, brat_data_dir)

    def load(self, data_generator, data_source=None):
        for guid, text, _, json_tags in data_generator(data_source):
            tagged_text = TaggedText(guid, text)
            for json_tag in json_tags:
                #  logger.info(f"json_tag: {json_tag}")
                entity_tag = RelTag().from_dict(json_tag)
                tagged_text.add_tag(entity_tag)
            self.tagged_text_list.append(tagged_text)
        return self

    def export_to_brat(self, brat_data_dir, max_pages=10):
        from .brat import export_brat_files
        export_brat_files(self.tagged_text_list,
                          self.ner_labels,
                          self.ner_connections,
                          brat_data_dir,
                          max_pages=max_pages)

    def import_from_brat(self, brat_data_dir):
        from .brat import import_brat_files
        import_brat_files(self.tagged_text_list, brat_data_dir)

    def export_to_poplar(self, poplar_file, max_pages=100, start_page=0):
        from .poplar import save_poplar_file
        save_poplar_file(poplar_file,
                         self.ner_labels,
                         self.ner_connections,
                         start_page=start_page,
                         max_pages=max_pages)

    def import_from_poplar(self, poplar_file):
        from .poplar import poplar_data_generator
        return self.load(poplar_data_generator(poplar_file))

    def diff(self, another_ner_dataset):
        assert len(self) == len(another_ner_dataset)

        identical_tags_list = []
        a_only_tags_list = []
        b_only_tags_list = []
        for tagged_text_a, tagged_text_b in zip(self, another_ner_dataset):
            text_a, text_b = tagged_text_a.text, tagged_text_b.text
            # 被标注的文本必须相同，否则标注位置无意义
            assert text_a == text_b
            tags_a, tags_b = tagged_text_a.tags, tagged_text_b.tags
            tags_a = sorted(tags_a, key=lambda x: x.start)
            tags_b = sorted(tags_b, key=lambda x: x.start)

            identical_tags = []
            a_only_tags = []
            b_only_tags = []
            for tag_a in tags_a:
                found_a = False
                for tag_b in tags_b:
                    if tag_a == tag_b:
                        identical_tags.append(tag_a)
                        found_a = True
                        break
                if not found_a:
                    a_only_tags.append(tag_a)
            for tag_b in tags_b:
                found_b = False
                for tag_a in tags_a:
                    if tag_a == tag_b:
                        found_b = True
                        break
                if not found_b:
                    b_only_tags.append(tag_b)

            identical_tags_list.append(identical_tags)
            a_only_tags_list.append(a_only_tags)
            b_only_tags_list.append(b_only_tags)

        return identical_tags_list, a_only_tags_list, b_only_tags_list


def merge_spo_datasets(spo_dataset_list, min_dups=2):
    if len(spo_dataset_list) == 0:
        return None
    from copy import deepcopy
    merged_dataset = deepcopy(spo_dataset_list[0])
    if len(spo_dataset_list) > 1:
        for i, X in enumerate(zip(*spo_dataset_list)):
            spo_list_list = [x.spo_list for x in X]
            from ..utils import merge_entities
            new_spo_list = merge_entities(
                spo_list_list,
                key=lambda x: x.start * 1000 + len(x.mention),
                min_dups=min_dups)
            merged_dataset[i].spo_list = new_spo_list

    return merged_dataset


def spo_data_generator(train_file, dataset_name=None, predicate_labels=None):
    spo_dataset = SpoDataset(dataset_name, predicate_labels)
    spo_dataset.load_from_file(train_file)
    for tagged_text in tqdm(spo_dataset):
        guid = tagged_text.guid
        text = tagged_text.text
        spo_list = [x.to_dict() for x in tagged_text.spo_list]
        yield guid, text, None, spo_list


def load_spo_dataset(filename):
    spo_dataset = SpoDataset()
    spo_dataset.load_from_file(filename)
    return spo_dataset


if __name__ == '__main__':
    pass
