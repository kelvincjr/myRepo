#!/usr/bin/env python
# -*- coding: utf-8 -*-

# NER数据集持久化，与标注文件之间的互转。

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


# 打过标签的文本
@dataclass
class TaggedText:
    guid: str
    text: str
    tags: List[EntityTag] = field(default_factory=list)

    def clear(self):
        self.tags = []

    def to_dict(self):
        return {
            'guid': self.guid,
            'text': self.text,
            'tags': [x.to_dict() for x in self.tags]
        }

    def from_dict(self, dict_data):
        self.clear()
        if 'guid' in dict_data:
            self.guid = dict_data['guid']
        elif 'text' in dict_data:
            self.text = dict_data['text']
        elif 'tags' in dict_data:
            for x in dict_data['tags']:
                self.tags.append(EntityTag().from_dict(x))

        return self

    def add_tag(self, tag: EntityTag):
        self.tags.append(tag)


# 实体标签数据集
class NerDataset:
    def __init__(self, name=None, ner_labels=None, ner_connections=None):
        self.name = name
        self.ner_labels = ner_labels
        self.ner_connections = ner_connections
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
        self.tagged_text_list.extend(other_dataset.tagged_text_list)

    def save(self, filename: str, format="json"):
        """
        {'guid': '0000', 'text': "sample text", 'tags': [{'category': "实体类别1", start: 10, mention: "实体文本"}, ...]}
        """

        logger.debug(f"Save {filename} with format {format}")
        if format == "json":
            json_data = {x.guid: x.to_dict() for x in self.tagged_text_list}
            json.dump(json_data,
                      open(filename, 'w'),
                      ensure_ascii=False,
                      indent=2)
        elif format == 'lines':
            with open(filename, 'w') as wt:
                for tagged_text in tqdm(self.tagged_text_list,
                                        desc="Save tagged text list."):
                    data_dict = tagged_text.to_dict()
                    wt.write(f"{json.dumps(data_dict, ensure_ascii=False)}\n")
        else:
            raise ValueError(
                f"Bad format: {format}. Format must be one of ['json', 'lines']"
            )
        logger.warning(f"Saved {filename}")

    def load_from_file(self, filename: str):
        logger.info(f"Loading {filename}")
        try:
            lines = [json.loads(x) for x in open(filename, 'r').readlines()]
        except:
            logger.warning(
                f"{filename} is not standard style, try reviews style...")
            try:
                json_data = json.load(open(filename, 'r'))
                lines = [v for k, v in json_data.items()]
            except:
                logger.error(f"Unknown file style {filename}")
                raise Exception(f"Unknown file style {filename}")

        def data_generator(data_source=None):
            for json_data in lines:
                yield json_data['guid'], json_data['text'], None, json_data[
                    'tags']

        return self.load(data_generator)

    def load_from_brat_data(self, brat_data_dir):
        from .brat import brat_data_generator, get_brat_schemas
        ner_labels, ner_connections = get_brat_schemas(brat_data_dir)
        self.ner_labels = ner_labels
        self.ner_connections = ner_connections

        return self.load(brat_data_generator, brat_data_dir)

    def load(self, data_generator, data_source=None):
        categories = []
        for guid, text, _, json_tags in data_generator(data_source):
            tagged_text = TaggedText(guid, text)
            for json_tag in json_tags:
                entity_tag = EntityTag().from_dict(json_tag)
                tagged_text.add_tag(entity_tag)

                category = entity_tag.category
                if category not in categories:
                    categories.append(category)

            self.tagged_text_list.append(tagged_text)

        logger.info(f"Loaded categories: {categories}")
        if not self.ner_labels:
            self.ner_labels = categories

        return self

    def export_to_brat(self, brat_data_dir, max_pages=10):
        from .brat import export_brat_files
        export_brat_files(self.tagged_text_list,
                          self.ner_labels,
                          self.ner_connections,
                          brat_data_dir,
                          max_pages=max_pages)

    def export_to_poplar(self, poplar_file, max_pages=100, start_page=0):
        from .poplar import save_poplar_file
        save_poplar_file(self.tagged_text_list,
                         poplar_file,
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


def merge_ner_datasets(ner_dataset_list, min_dups=2):
    if len(ner_dataset_list) == 0:
        return None
    from copy import deepcopy
    merged_dataset = deepcopy(ner_dataset_list[0])
    if len(ner_dataset_list) > 1:
        for i, X in enumerate(
                tqdm(zip(*ner_dataset_list), desc="Merge ner datasets")):
            tags_list = [x.tags for x in X]
            from ..utils import merge_entities
            new_tags = merge_entities(
                tags_list,
                key=lambda x: x.start * 1000 + len(x.mention),
                min_dups=min_dups)
            merged_dataset[i].tags = new_tags

    return merged_dataset


def mix_ner_datasets(ner_dataset_list):
    """
    以ner_data_list[0]为基准，只保留实体文本区间不重叠的部分
    """
    if len(ner_dataset_list) == 0:
        return None
    from copy import deepcopy
    mixed_dataset = deepcopy(ner_dataset_list[0])
    for i, X in enumerate(tqdm(zip(*ner_dataset_list),
                               desc="Mix ner datasets")):
        mixed_tags = mixed_dataset[i].tags
        #  logger.info(f"mixed_tags: {mixed_tags}")
        tags_list = [x.tags for x in X]
        for tags in tags_list:
            for tag in tags:
                c = tag.category
                s = tag.start
                m = tag.mention
                e = s + len(m) - 1
                overlap = False
                for tag0 in mixed_tags:
                    #  logger.info(f"tag0: {tag0}")
                    c0 = tag0.category
                    s0 = tag0.start
                    m0 = tag0.mention
                    e0 = s0 + len(m0) - 1
                    if s >= s0 and s <= e0:
                        overlap = True
                        break
                    if e >= s0 and e <= e0:
                        overlap = True
                        break
                    if s0 >= s and s0 <= e:
                        overlap = True
                        break
                    if e0 >= s and e0 <= e:
                        overlap = True
                        break
                if not overlap:
                    mixed_tags.append(EntityTag(c, s, m))
        mixed_tags = sorted(mixed_tags, key=lambda x: x.start)
        mixed_dataset[i].tags = mixed_tags
    return mixed_dataset


def ner_data_generator(train_file,
                       dataset_name=None,
                       ner_labels=None,
                       ner_connections=None):
    ner_dataset = NerDataset("kgcs_entities", ner_labels, ner_connections)
    ner_dataset.load_from_file(train_file)
    for tagged_text in tqdm(ner_dataset):
        guid = tagged_text.guid
        text = tagged_text.text
        tags = [x.to_dict() for x in tagged_text.tags]
        #  logger.debug(f"tags: {tags}")
        yield guid, text, None, tags


def load_ner_dataset(filename):
    ner_dataset = NerDataset()
    ner_dataset.load_from_file(filename)
    return ner_dataset


if __name__ == '__main__':
    pass
