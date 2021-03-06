#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, random, copy, json
from tqdm import tqdm
from loguru import logger
from theta.utils import seg_generator
from dataclasses import dataclass, field
from typing import List
#import mlflow

# ----------------------
# 实体标注规范
#  {
#      'guid':
#      "0",
#      'text':
#      'xxxxxxx',
#      'entities': [{
#          'category': category,
#          'start': start,
#          'end': end,
#          'mention': mention
#      }]
#  }


class DataClassBase:
    def to_dict(self):
        #  return self.__dict__
        dict_obj = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            if isinstance(v, List):
                dict_obj[k] = [
                    x.to_dict() if isinstance(x, DataClassBase) else x
                    for x in v
                ]
            else:
                dict_obj[k] = v
        return dict_obj

    def from_dict(self, dict_data):
        for k in self.__dict__.keys():
            if k in dict_data:
                setattr(self, k, dict_data[k])


@dataclass(unsafe_hash=True)
class Entity(DataClassBase):
    category: str = field(default_factory=str)
    start: int = field(default_factory=int)
    end: int = field(default_factory=int)
    mention: str = field(default_factory=str)


@dataclass
class LabeledText(DataClassBase):
    guid: str = field(default_factory=str)
    text: str = field(default_factory=str)
    entities: List[Entity] = field(default_factory=list)

    def add_entity(self, category, start, end):
        self.entities.append(
            Entity(category=category,
                   start=start,
                   end=end,
                   mention=self.text[start:end + 1]))

    def offset(self, text_offset):
        for entity in self.entities:
            entity.start += text_offset
            entity.end += text_offset

    def get_annotated_text(self, seg_len, seg_backoff):
        #  debug = False
        #  if self.guid == '449':
        #      debug = True

        max_text_length = seg_len - seg_backoff

        text = self.text[:max_text_length]
        #  if debug:
        #      logger.debug(f"text: <{text}>")
        annotated_text = ""
        p0 = 0
        for entity in self.entities:
            x0 = entity.start
            x1 = entity.end
            if x1 >= max_text_length:
                continue
            annotated_text += text[p0:x0]
            #  if debug:
            #      logger.debug(
            #          f"annotated_text(p0:{p0},x0:{x0},x1:{x1}): {annotated_text}"
            #      )
            annotated_text += f"【{entity.mention}】"
            #  if debug:
            #      logger.debug(
            #          f"annotated_text(p0:{p0},x0:{x0},x1:{x1}): {annotated_text}"
            #      )
            p0 = x1 + 1
        annotated_text += text[p0:]
        #  if debug:
        #      logger.debug(f"annotated_text(p0:{p0}): {annotated_text}")

        return annotated_text

    #  def to_json(self):
    #      json_entities = [x.to_dict() for x in self.entities]
    #      json_data = {
    #          'guid': self.guid,
    #          'text': self.text,
    #          'entities': json_entities
    #      }
    #      return json_data

    #  def from_json(self, json_data):
    #      self.guid = json_data['guid']
    #      self.text = json_data['text']
    #      self.entities = []
    #      for x in json_data['entities']:
    #          entity = Entity()
    #          entity.from_dict(x)
    #          self.entities.append(entity)


def get_ner_preds_reviews(preds, examples, seg_len, seg_backoff):
    reviews = {}
    category_mentions = {}
    for json_d, input_example in tqdm(zip(preds, examples)):
        guid = input_example.guid
        text_offset = input_example.text_offset
        text = input_example.text_a
        entities = json_d['entities']

        labeled_text = LabeledText(guid, text)
        for c, x0, x1 in entities:
            labeled_text.add_entity(c, x0, x1)
            if c not in category_mentions:
                category_mentions[c] = set()
            category_mentions[c].add(labeled_text.entities[-1].mention)

        annotated_text = labeled_text.get_annotated_text(seg_len, seg_backoff)

        labeled_text.offset(text_offset)
        json_data = labeled_text.to_dict()

        if guid not in reviews:
            reviews[guid] = {
                'guid': guid,
                'text': "",
                'annotated_text': "",
                'entities': []
            }
        reviews[guid]['text'] += text[:seg_len - seg_backoff]
        reviews[guid]['annotated_text'] += annotated_text
        for x in json_data['entities']:
            if x not in reviews[guid]['entities']:
                reviews[guid]['entities'].append(x)
        reviews[guid]['entities'] = sorted(reviews[guid]['entities'],
                                           key=lambda x: x['start'])

    return reviews, category_mentions


def save_ner_preds(args, preds, test_examples):
    reviews, category_mentions = get_ner_preds_reviews(preds, test_examples,
                                                       args.seg_len,
                                                       args.seg_backoff)

    reviews_file = f"{args.latest_dir}/{args.dataset_name}_reviews_{args.local_id}.json"

    json.dump(reviews, open(reviews_file, 'w'), ensure_ascii=False, indent=2)
    logger.info(f"Reviews file: {reviews_file}")

    category_mentions_file = f"{args.latest_dir}/{args.dataset_name}_category_mentions_{args.local_id}.txt"
    num_categories = len(category_mentions)
    num_mentions = 0
    with open(category_mentions_file, 'w') as wt:
        for c, mentions in category_mentions.items():
            for m in mentions:
                wt.write(f"{c}\t{m}\n")
                num_mentions += 1
    logger.info(
        f"Total {num_categories} categories and {num_mentions} mentions saved to {category_mentions_file}"
    )

    #  ----- Tracking -----
    if args.do_experiment:
        mlflow.log_param(f"{args.dataset_name}_reviews_file", reviews_file)
        mlflow.log_artifact(reviews_file, args.artifact_path)
        mlflow.log_param(f"{args.dataset_name}_category_mentions_file",
                         category_mentions_file)
        mlflow.log_artifact(category_mentions_file, args.artifact_path)

    return reviews_file, category_mentions_file


def augement_entities(all_text_entities, labels_map, num_augements):
    aug_tokens = []
    for i, (guid, text, entities) in enumerate(
            tqdm(all_text_entities, desc=f"Augement {num_augements}X")):

        #  print(f"-------------------{json_file}--------------------")
        #  print(text)
        #  print(entities)
        #  for entity in entities:
        #      s = entity['start_pos']
        #      e = entity['end_pos']
        #      print(f"{entity['label_type']}: {text[s:e]}")
        #  print("----------------------------------------")
        if entities:
            for ai in range(num_augements):
                e_idx = random.randint(0, len(entities) - 1)
                entity = entities[e_idx]

                label_type = entity['category']
                s = entity['start']
                e = entity['end']

                labels = labels_map[label_type]
                idx = random.randint(0, len(labels) - 1)
                new_entity_text = labels[idx]

                text = text[:s] + new_entity_text + text[e + 1:]

                assert len(new_entity_text) >= 0
                delta = len(new_entity_text) - (e + 1 - s)

                entity['end'] = entity['start'] + len(new_entity_text) - 1
                entity['mention'] = new_entity_text

                assert text[entity['start']:entity['end'] +
                            1] == new_entity_text

                for n, e in enumerate(entities):
                    if n > e_idx:
                        e['start'] += delta
                        e['end'] += delta

                aug_tokens.append(
                    (f"{guid}-a{ai}", text, copy.deepcopy(entities)))

    #  for guid, text, entities in aug_tokens:
    #      text_a = text
    #      for entity in entities:
    #          logger.debug(f"{guid}: text_a: {text_a}")
    #          logger.debug(
    #              f"text_a[entity['start']:entity['end']]: {text_a[entity['start']:entity['end']]}"
    #          )
    #          logger.debug(
    #              f"mention {entity['mention']} in {text_a.find(entity['mention'])}"
    #          )
    #          logger.debug(f"entity: {entity}")
    #          assert text_a[entity['start']:entity['end']] == entity[
    #              'mention']

    return aug_tokens


def data_seg_generator(lines,
                       ner_labels,
                       seg_len=0,
                       seg_backoff=0,
                       num_augements=0,
                       allow_overlap=False):
    assert seg_backoff >= 0 and seg_backoff <= int(seg_len * 3 / 4)
    all_text_entities = []
    labels_map = {}

    num_overlap = 0
    for i, s in enumerate(tqdm(lines)):
        #  logger.debug(f"s: {s}")
        guid = str(i)
        text = s['text'].strip()
        entities = s['entities']

        new_entities = []
        used_span = []
        #  logger.debug(f"entities: {entities}")
        entities = sorted(entities, key=lambda e: e.start)
        for entity in entities:
            if entity.category not in ner_labels:
                continue
            entity.mention = text[entity.start:entity.end + 1]
            s = entity.start
            e = entity.end

            overlap = False
            for us in used_span:
                if s >= us[0] and s <= us[1]:
                    overlap = True
                    break
                if e >= us[0] and e <= us[1]:
                    overlap = True
                    break
            if overlap:
                num_overlap += 1
                if not allow_overlap:
                    #  logger.warning(
                    #      f"Overlap! {i} mention: {entity.mention}({s}:{e}), used_span: {used_span}"
                    #  )
                    continue
            used_span.append((s, e))

            new_entities.append(entity)
        entities = new_entities

        seg_offset = 0
        #  if seg_len <= 0:
        #      seg_len = max_seq_length

        for (seg_text, ), text_offset in seg_generator((text, ), seg_len,
                                                       seg_backoff):
            text_a = seg_text

            assert text_offset == seg_offset

            seg_start = seg_offset
            seg_end = seg_offset + min(seg_len, len(seg_text))
            labels = [(x.category, x.start - seg_offset, x.end - seg_offset)
                      for x in entities
                      if x.start >= seg_offset and x.end < seg_end]

            # 没有标注存在的文本片断不用于训练
            if labels:
                yield guid, text_a, None, labels, seg_offset

                if num_augements > 0:
                    seg_entities = [
                        {
                            'start': x.start - seg_offset,
                            'end': x.end - seg_offset,
                            'category': x.category,
                            'mention': x.mention
                        } for x in entities
                        if x.start >= seg_offset and x.end < seg_end
                    ]
                    all_text_entities.append((guid, text_a, seg_entities))

                    for entity in seg_entities:
                        label_type = entity['category']
                        s = entity['start']  # - seg_offset
                        e = entity['end']  #- seg_offset
                        #  print(s, e)
                        assert e >= s
                        #  logger.debug(
                        #      f"seg_start: {seg_start}, seg_end: {seg_end}, seg_offset: {seg_offset}"
                        #  )
                        #  logger.debug(f"s: {s}, e: {e}")
                        assert s >= 0 and e < len(seg_text)
                        #  if s >= len(seg_text) or e >= len(seg_text):
                        #      continue

                        entity_text = seg_text[s:e + 1]
                        #  print(label_type, entity_text)

                        assert len(entity_text) > 0
                        if label_type not in labels_map:
                            labels_map[label_type] = []
                        labels_map[label_type].append(entity_text)

            seg_offset += seg_len - seg_backoff

    logger.warning(f"num_overlap: {num_overlap}")

    if num_augements > 0:
        aug_tokens = augement_entities(all_text_entities,
                                       labels_map,
                                       num_augements=num_augements)
        for guid, text, entities in aug_tokens:
            text_a = text
            for entity in entities:
                #  logger.debug(f"text_a: {text_a}")
                #  logger.debug(
                #      f"text_a[entity['start']:entity['end']]: {text_a[entity['start']:entity['end']]}"
                #  )
                #  logger.debug(
                #      f"mention {entity['mention']} in {text_a.find(entity['mention'])}"
                #  )
                #  logger.debug(f"entity: {entity}")
                assert text_a[entity['start']:entity['end'] +
                              1] == entity['mention']
            labels = [
                (entity['category'], entity['start'], entity['end'])
                for entity in entities if entity['end'] < (
                    min(len(text_a), seg_len) if seg_len > 0 else len(text_a))
            ]
            yield guid, text_a, None, labels, 0


class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels, text_offset=0):
        self.guid = guid
        self.text_a = text_a
        self.labels = labels
        self.text_offset = text_offset

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def load_ner_examples(data_generator, examples_file, seg_len=0, seg_backoff=0):

    examples = []

    for guid, text_a, _, labels in data_generator(examples_file):
        assert text_a is not None
        for (seg_text_a, ), text_offset in seg_generator((text_a, ), seg_len,
                                                         seg_backoff):
            examples.append(
                InputExample(guid=guid,
                             text_a=seg_text_a,
                             labels=labels,
                             text_offset=text_offset))
    logger.info(f"Loaded {len(examples)} examples.")

    return examples


def load_ner_labeled_examples(lines,
                              ner_labels,
                              seg_len=0,
                              seg_backoff=0,
                              num_augements=0,
                              allow_overlap=False):

    examples = []
    for guid, text_a, text_b, labels, text_offset in data_seg_generator(
            lines,
            ner_labels,
            seg_len=seg_len,
            seg_backoff=seg_backoff,
            num_augements=num_augements,
            allow_overlap=allow_overlap):
        assert text_a is not None
        examples.append(
            InputExample(guid=guid,
                         text_a=text_a,
                         labels=labels,
                         text_offset=text_offset))

    logger.info(f"Loaded {len(examples)} examples.")
    return examples


def show_ner_datainfo(ner_labels, train_data_generator, train_file,
                      test_data_generator, test_file):
    from collections import defaultdict
    label_counts = defaultdict(int)

    label_examples = defaultdict(int)

    train_lengths = []
    for guid, text_a, _, labels in train_data_generator(train_file):
        entity_examples = defaultdict(int)
        train_lengths.append(len(text_a))
        for entity in labels:
            c = entity.category
            entity_examples[c] = 1
            label_counts[c] += 1
            if c not in ner_labels:
                ner_labels.append(c)
        for c, n in entity_examples.items():
            label_examples[c] += n

    test_lengths = [
        len(text_a)
        for guid, text_a, _, labels in test_data_generator(test_file)
    ]

    logger.info(f"****** ner_labels ******")
    logger.info(f"{len(ner_labels)} labels: {ner_labels}")
    logger.info(f"label_counts: {label_counts}")
    logger.info(
        f"num train examples: {len(train_lengths)}, num test examples: {len(test_lengths)}"
    )
    logger.info(f"label_examples: {label_examples}")
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


def to_poplar(args, poplar_data_file, pages, ner_labels, ner_connections,
              start_page, max_pages):
    poplar_json = {
        "content": "",
        "labelCategories": [],
        "labels": [],
        "connectionCategories": [],
        "connections": []
    }

    poplar_colorset = [
        '#007bff', '#17a2b8', '#28a745', '#fd7e14', '#e83e8c', '#dc3545',
        '#20c997', '#ffc107', '#007bff'
    ]
    label2id = {x: i for i, x in enumerate(ner_labels)}
    label_categories = poplar_json['labelCategories']
    for _id, x in enumerate(ner_labels):
        label_categories.append({
            "id":
            _id,
            "text":
            x,
            "color":
            poplar_colorset[label2id[x] % len(poplar_colorset)],
            "borderColor":
            "#cccccc"
        })

    connection_categories = poplar_json['connectionCategories']
    for _id, _text in enumerate(ner_connections):
        connection_categories.append({'id': _id, 'text': _text})

    poplar_labels = poplar_json['labels']
    poplar_content = ""
    eid = 0
    num_pages = 0
    page_offset = 0

    for guid, text in pages:

        if num_pages < start_page:
            num_pages += 1
            continue

        page_head = f"\n-------------------- {guid} Begin --------------------\n\n"
        page_tail = f"\n-------------------- {guid} End --------------------\n\n"
        poplar_content += page_head + f"{text}" + page_tail

        num_pages += 1
        if num_pages - start_page >= max_pages:
            break

        page_offset = len(poplar_content)

    poplar_json["content"] = poplar_content
    poplar_json['labels'] = poplar_labels

    json.dump(poplar_json,
              open(poplar_data_file, 'w'),
              ensure_ascii=False,
              indent=2)
    logger.info(f"Saved {poplar_data_file}")

def to_sampling_poplar(args,
                    train_data_generator,
                    ner_labels,
                    ner_connections,
                    start_page=0,
                    max_pages=100):
    poplar_json = {
        "content": "",
        "labelCategories": [],
        "labels": [],
        "connectionCategories": [],
        "connections": []
    }

    poplar_colorset = [
        '#007bff', '#17a2b8', '#28a745', '#fd7e14', '#e83e8c', '#dc3545',
        '#20c997', '#ffc107', '#007bff'
    ]
    label2id = {x: i for i, x in enumerate(ner_labels)}
    label_categories = poplar_json['labelCategories']
    for _id, x in enumerate(ner_labels):
        label_categories.append({
            "id":
            _id,
            "text":
            x,
            "color":
            poplar_colorset[label2id[x] % len(poplar_colorset)],
            "borderColor":
            "#cccccc"
        })

    connection_categories = poplar_json['connectionCategories']
    for _id, _text in enumerate(ner_connections):
        connection_categories.append({'id': _id, 'text': _text})

    poplar_labels = poplar_json['labels']
    poplar_content = ""
    eid = 0
    num_pages = 0
    page_offset = 0
    for guid, text, _, entities in train_data_generator(args.train_file):
        if num_pages < start_page:
            num_pages += 1
            continue

        page_head = f"\n-------------------- {guid} Begin --------------------\n\n"
        page_tail = f"\n-------------------- {guid} End --------------------\n\n"
        poplar_content += page_head + f"{text}" + page_tail

        for entity in entities:
            poplar_labels.append({
                "id":
                eid,
                "categoryId":
                label2id[entity.category],
                "startIndex":
                page_offset + len(page_head) + entity.start,
                "endIndex":
                page_offset + len(page_head) + entity.end + 1,
            })
            eid += 1

        num_pages += 1
        if num_pages - start_page >= max_pages:
            break

        page_offset = len(poplar_content)

    poplar_json["content"] = poplar_content
    poplar_json['labels'] = poplar_labels

    if not os.path.exists("./poplar"):
        os.makedirs("./poplar")
    poplar_data_file = f"./poplar/poplar_sampling_data_{args.local_id}_{max_pages}_{start_page}.json"
    json.dump(poplar_json,
              open(poplar_data_file, 'w'),
              ensure_ascii=False,
              indent=2)
    logger.info(f"Saved {poplar_data_file}")

def to_train_poplar(args,
                    train_data_generator,
                    ner_labels,
                    ner_connections,
                    start_page=0,
                    max_pages=100):
    poplar_json = {
        "content": "",
        "labelCategories": [],
        "labels": [],
        "connectionCategories": [],
        "connections": []
    }

    poplar_colorset = [
        '#007bff', '#17a2b8', '#28a745', '#fd7e14', '#e83e8c', '#dc3545',
        '#20c997', '#ffc107', '#007bff'
    ]
    label2id = {x: i for i, x in enumerate(ner_labels)}
    label_categories = poplar_json['labelCategories']
    for _id, x in enumerate(ner_labels):
        label_categories.append({
            "id":
            _id,
            "text":
            x,
            "color":
            poplar_colorset[label2id[x] % len(poplar_colorset)],
            "borderColor":
            "#cccccc"
        })

    connection_categories = poplar_json['connectionCategories']
    for _id, _text in enumerate(ner_connections):
        connection_categories.append({'id': _id, 'text': _text})

    poplar_labels = poplar_json['labels']
    poplar_content = ""
    eid = 0
    num_pages = 0
    page_offset = 0
    for guid, text, _, entities in train_data_generator(args.train_file):
        if num_pages < start_page:
            num_pages += 1
            continue

        page_head = f"\n-------------------- {guid} Begin --------------------\n\n"
        page_tail = f"\n-------------------- {guid} End --------------------\n\n"
        poplar_content += page_head + f"{text}" + page_tail

        for entity in entities:
            poplar_labels.append({
                "id":
                eid,
                "categoryId":
                label2id[entity.category],
                "startIndex":
                page_offset + len(page_head) + entity.start,
                "endIndex":
                page_offset + len(page_head) + entity.end + 1,
            })
            eid += 1

        num_pages += 1
        if num_pages - start_page >= max_pages:
            break

        page_offset = len(poplar_content)

    poplar_json["content"] = poplar_content
    poplar_json['labels'] = poplar_labels

    if not os.path.exists("./poplar"):
        os.makedirs("./poplar")
    poplar_data_file = f"./poplar/poplar_train_data_{args.local_id}_{max_pages}_{start_page}.json"
    json.dump(poplar_json,
              open(poplar_data_file, 'w'),
              ensure_ascii=False,
              indent=2)
    logger.info(f"Saved {poplar_data_file}")


def to_reviews_poplar(args,
                      ner_labels,
                      ner_connections,
                      start_page=0,
                      max_pages=100):
    poplar_json = {
        "content": "",
        "labelCategories": [],
        "labels": [],
        "connectionCategories": [],
        "connections": []
    }

    poplar_colorset = [
        '#007bff', '#17a2b8', '#28a745', '#fd7e14', '#e83e8c', '#dc3545',
        '#20c997', '#ffc107', '#007bff'
    ]
    label2id = {x: i for i, x in enumerate(ner_labels)}
    label_categories = poplar_json['labelCategories']
    for _id, x in enumerate(ner_labels):
        label_categories.append({
            "id":
            _id,
            "text":
            x,
            "color":
            poplar_colorset[label2id[x] % len(poplar_colorset)],
            "borderColor":
            "#cccccc"
        })

    connection_categories = poplar_json['connectionCategories']
    for _id, _text in enumerate(ner_connections):
        connection_categories.append({'id': _id, 'text': _text})

    poplar_labels = poplar_json['labels']
    poplar_content = ""
    num_pages = 0
    page_offset = 0

    reviews_file = f"{args.latest_dir}/{args.dataset_name}_reviews_{args.local_id}.json"
    reviews = json.load(open(reviews_file, 'r'))

    eid = 0
    for guid, json_data in reviews.items():
        if num_pages < start_page:
            num_pages += 1
            continue
        text = json_data['text']

        page_head = f"\n-------------------- {guid} Begin --------------------\n\n"
        page_tail = f"\n-------------------- {guid} End --------------------\n\n"
        poplar_content += page_head + f"{text}" + page_tail

        for json_entity in json_data['entities']:
            category = json_entity['category']
            mention = json_entity['mention']
            start_char = json_entity['start']
            end_char = json_entity['end']

            poplar_labels.append({
                "id":
                eid,
                "categoryId":
                label2id[category],
                "startIndex":
                page_offset + len(page_head) + start_char,
                "endIndex":
                page_offset + len(page_head) + end_char + 1,
            })
            eid += 1

        num_pages += 1
        if num_pages - start_page >= max_pages:
            break

        page_offset = len(poplar_content)

    poplar_json["content"] = poplar_content
    poplar_json['labels'] = poplar_labels

    if not os.path.exists("./poplar"):
        os.makedirs("./poplar")
    poplar_data_file = f"./poplar/poplar_reviews_data_{args.local_id}_{max_pages}_{start_page}.json"
    json.dump(poplar_json,
              open(poplar_data_file, 'w'),
              ensure_ascii=False,
              indent=2)
    logger.info(f"Saved {poplar_data_file}")
