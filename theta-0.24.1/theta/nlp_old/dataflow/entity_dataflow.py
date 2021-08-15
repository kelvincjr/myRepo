#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List

from loguru import logger
from tqdm import tqdm

#from .dataflow import RNGDataFlow


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

    def __repr__(self):
        return f"{self.to_dict()}"


# 打过标签的文本
@dataclass
class TaggedText:
    guid: str
    text: str
    tags: List[EntityTag] = field(default_factory=list)
    text_offset: int = 0

    def clear(self):
        self.tags = []

    def offset(self, text_offset):
        self.text_offset += text_offset
        for tag in self.tags:
            tag.start += text_offset

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

    def __repr__(self):
        #  return f"{self.guid}: {self.text[:50]} | {self.text_offset} | {self.tags}"
        return f"guid: {self.guid}, text: {self.text[:50]}... | text_offset: {self.text_offset} | tags: {self.tags}"


#class BaseDataFlow(RNGDataFlow):
class BaseDataFlow(object):
    def __init__(self, data_list=None):
        super(BaseDataFlow, self).__init__()

        self._data_list = data_list if data_list is not None else []
        self._shuffle_idxs = []

    def __len__(self):
        return len(self._data_list)

    def __iter__(self):
        if len(self._shuffle_idxs) != self.__len__():
            self._shuffle_idxs = list(range(self.__len__()))

        for k in self._shuffle_idxs:
            yield self._data_list[k]

    def shuffle(self):
        self._shuffle_idxs = list(range(self.__len__()))
        #self.rng.shuffle(self._shuffle_idxs)

    def __getitem__(self, idx):
        return self._data_list[idx]

    def reset_state(self):
        #super(BaseDataFlow, self).reset_state()
        pass

    def clean(self):
        self._data_list = []
        self.reset_state()

    def append(self, data):
        self._data_list.append(data)

    def extend(self, other_dataset):
        self._data_list.extend(other_dataset._data_list)


class EntityDataFlow(BaseDataFlow):
    def __init__(self,
                 data_file=None,
                 data_generator=None,
                 data_list=None,
                 entity_labels=None):
        super(EntityDataFlow, self).__init__(data_list=data_list)

        self.entity_labels = entity_labels
        self.data_generator = data_generator
        self.data_file = data_file

        if data_list is None:
            if self.data_generator is not None or self.data_file is not None:
                self.load()

    def info(self):
        logger.info(f"Entity labels: {self.entity_labels}")
        logger.info(f"data_file: {self.data_file}")
        logger.info(f"Total: {self.__len__()}")
        for i in range(5):
            logger.info(f"{self.__getitem__(i)}")

    # ------------------------------ Load ------------------------------
    def load(self):
        if self.data_generator is not None:
            self.load_from_generator(self.data_generator)
        elif self.data_file is not None:
            self.load_from_file(self.data_file)
        else:
            raise ValueError(
                f"Either data_generator or data_file must be set.")

    def load_from_generator(self, data_generator, data_source=None):
        self.clean()

        self.data_generator = data_generator

        categories = []

        g = self.data_generator(data_source) if callable(
            self.data_generator) else self.data_generator
        for guid, text, _, json_tags in g:
            tagged_text = TaggedText(guid, text)
            if json_tags:
                for json_tag in json_tags:
                    entity_tag = EntityTag().from_dict(json_tag)
                    tagged_text.add_tag(entity_tag)

                    category = entity_tag.category
                    if category not in categories:
                        categories.append(category)

            self.append(tagged_text)

        logger.info(f"Loaded categories: {categories}")
        logger.info(f"Loaded {self.__len__()} samples.")
        if not self.entity_labels:
            self.entity_labels = categories

        return self

    def load_from_file(self, data_file: str):
        self.clean()

        self.data_file = data_file

        logger.info(f"Loading {data_file}")
        try:
            lines = [json.loads(x) for x in open(data_file, 'r').readlines()]
        except:
            logger.warning(
                f"{data_file} is not standard format, try reviews format...")
            try:
                json_data = json.load(open(data_file, 'r'))
                lines = [v for k, v in json_data.items()]
            except:
                logger.error(f"Unknown file format {data_file}")
                raise ValueError(f"Unknown file format {data_file}")

        def data_generator(data_source=None):
            for json_data in lines:
                yield json_data['guid'], json_data['text'], None, json_data[
                    'tags']

        return self.load_from_generator(data_generator)

    # ------------------------------ Save ------------------------------
    def save(self, filename: str, format="json"):
        """
        {'guid': '0000', 'text': "sample text", 'tags': [{'category': "实体类别1", start: 10, mention: "片断文本"}, ...]}
        """

        if format == "json":
            json_data = {x.guid: x.to_dict() for x in self}
            json.dump(json_data,
                      open(filename, 'w'),
                      ensure_ascii=False,
                      indent=2)
        elif format == 'lines':
            with open(filename, 'w') as wt:
                for tagged_text in tqdm(self, desc="Save tagged text list."):
                    data_dict = tagged_text.to_dict()
                    wt.write(f"{json.dumps(data_dict, ensure_ascii=False)}\n")
        else:
            raise ValueError(
                f"Bad format: {format}. Format must be one of ['json', 'lines']"
            )
        logger.warning(f"Saved {filename}")

    # ------------------------------ Brat data ------------------------------
    def import_from_brat_data(self, brat_data_dir):
        from .brat import brat_data_generator, get_brat_schemas
        entity_labels, relation_labels = get_brat_schemas(brat_data_dir)
        self.entity_labels = entity_labels

        return self.load_from_generator(brat_data_generator, brat_data_dir)

    def export_to_brat_data(self, brat_data_dir, max_pages=10):
        from .brat import export_brat_files
        export_brat_files(self._data_list,
                          self.entity_labels,
                          None,
                          brat_data_dir,
                          max_pages=max_pages)

    # ------------------------------ Poplar data ------------------------------
    def import_from_poplar_data(self, poplar_file):
        from .poplar import poplar_data_generator
        return self.load_from_generator(poplar_data_generator, poplar_file)

    def export_to_poplar_data(self, poplar_file, max_pages=100, start_page=0):
        from .poplar import save_poplar_file
        save_poplar_file(self._data_list,
                         poplar_file,
                         self.entity_labels,
                         None,
                         start_page=start_page,
                         max_pages=max_pages)

    # ------------------------------ Utils ------------------------------
    def diff(self, another_dataflow):
        assert len(self) == len(another_dataflow)

        identical_tags_list = []
        a_only_tags_list = []
        b_only_tags_list = []
        for tagged_text_a, tagged_text_b in zip(self, another_dataflow):
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


def merge_entity_dataflows(entity_dataflow_list, min_dups=2):
    assert entity_dataflow_list is not None
    assert len(entity_dataflow_list) > 0

    def remove_duplicate_entities(R):
        new_R = []
        for i, r in enumerate(R):
            found_dup = False
            for r1 in R[i + 1:]:
                if r == r1:
                    found_dup = True
                    break
            if found_dup:
                continue
            new_R.append(r)
        return new_R

    def is_uniform(entities):
        for i, x in enumerate(entities):
            for x1 in entities[i:]:
                if x != x1:
                    return False
        return True

    def merge_entities(entities_list, key=None, min_dups=2):
        new_X = []
        X = [x for z in entities_list for x in z]
        #  logger.info(f"X: {X}")
        if key:
            X = sorted(X, key=key)

        for i, x in enumerate(X[:1 - min_dups]):
            if is_uniform(X[i:i + min_dups]):
                new_X.append(deepcopy(x))
        new_X = remove_duplicate_entities(new_X)
        return new_X

    merged_entity_dataflow = deepcopy(entity_dataflow_list[0])
    if len(entity_dataflow_list) > 1:
        for i, X in enumerate(
                tqdm(zip(*entity_dataflow_list),
                     desc="Merge entity dataflows")):
            tags_list = [x.tags for x in X]
            new_tags = merge_entities(
                tags_list,
                key=lambda x: x.start * 1000 + len(x.mention),
                min_dups=min_dups)
            merged_entity_dataflow[i].tags = new_tags

    return merged_entity_dataflow


def mix_entity_dataflows(entity_dataflow_list):
    """
    以entity_dataflow_list[0]为基准，只保留实体文本区间不重叠的部分
    """
    if len(entity_dataflow_list) == 0:
        return None
    mixed_entity_dataflow = deepcopy(entity_dataflow_list[0])
    for i, X in enumerate(
            tqdm(zip(*entity_dataflow_list), desc="Mix ner datasets")):
        mixed_tags = mixed_entity_dataflow[i].tags
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
        mixed_entity_dataflow[i].tags = mixed_tags
    return mixed_entity_dataflow


if __name__ == '__main__':
    # from theta.nlp.dataflow import EntityDataFlow
    df = EntityDataFlow(
        data_file="../../tutorials/cluener/cluener_train_data.json")
    df.info()
