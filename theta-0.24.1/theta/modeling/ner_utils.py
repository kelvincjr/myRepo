#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import json
import os
import random
from dataclasses import dataclass, field
from typing import List

from loguru import logger
from tqdm import tqdm

from theta.utils import seg_generator

from ..utils import DataClassBase

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

        #  logger.info(
        #      f"guid: {guid}, offset: {text_offset}, entities: {entities}")

        labeled_text = LabeledText(guid, text)
        for c, x0, x1 in entities:
            if x0 >= len(text):
                logger.info(f"x0({x0}) >= len(text)({len(text)})")
                logger.warning(
                    f"pos overflow [{guid}]:({c},{x0},{x1}) text: ({len(text)}) {text}"
                )
            if x0 < 0 or x1 < 0 or x0 > x1:
                logger.warning(f"Invalid x0({x0}) and x1({x1})")

            labeled_text.add_entity(c, x0, x1)
            if c not in category_mentions:
                category_mentions[c] = set()
            category_mentions[c].add(labeled_text.entities[-1].mention)

        annotated_text = labeled_text.get_annotated_text(seg_len, seg_backoff)

        labeled_text.offset(text_offset)
        json_data = labeled_text.to_dict()
        json_entities = json_data['entities']

        if guid not in reviews:
            reviews[guid] = {
                'guid': guid,
                'text': "",
                'annotated_text': "",
                'tags': []
            }
        reviews[guid]['text'] += text[:seg_len - seg_backoff]
        reviews[guid]['annotated_text'] += annotated_text
        reviews_tags = reviews[guid]['tags']
        for x in json_entities:
            if x not in reviews_tags:
                c = x['category']
                s = int(x['start'])
                m = x['mention']
                #  e = s + len(m) - 1
                #  m = text[s:e + 1]

                #  if not m:
                #      logger.warning(
                #          f"Mention is None. guid: {guid}, text: {text}, c: {c}, s: {s}, m: {m}"
                #      )
                reviews_tags.append({'category': c, 'start': s, 'mention': m})

        reviews_tags = sorted(reviews_tags, key=lambda x: x['start'])
        from ..utils import remove_duplicate_entities
        reviews_tags = remove_duplicate_entities(reviews_tags)
        reviews[guid]['tags'] = reviews_tags
        #  logger.info(f"guid: {guid}, tags: {reviews[guid]['tags']}")

    return reviews, category_mentions


def save_ner_preds(args, preds, test_examples):
    reviews, category_mentions = get_ner_preds_reviews(preds, test_examples,
                                                       args.seg_len,
                                                       args.seg_backoff)

    #  reviews_file = f"{args.latest_dir}/{args.dataset_name}_reviews_{args.local_id}.json"
    reviews_file = args.reviews_file

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
    #  if args.do_experiment:
    #      mlflow.log_param(f"{args.dataset_name}_reviews_file", reviews_file)
    #      mlflow.log_artifact(reviews_file, args.artifact_path)
    #      mlflow.log_param(f"{args.dataset_name}_category_mentions_file",
    #                       category_mentions_file)
    #      mlflow.log_artifact(category_mentions_file, args.artifact_path)

    return reviews_file, category_mentions_file


def augment_entities(all_text_entities, labels_map, num_augments):
    aug_tokens = []
    for i, (guid, text, entities) in enumerate(
            tqdm(all_text_entities, desc=f"Augement {num_augments}X")):

        #  print(f"-------------------{json_file}--------------------")
        #  print(text)
        #  print(entities)
        #  for entity in entities:
        #      s = entity['start_pos']
        #      e = entity['end_pos']
        #      print(f"{entity['label_type']}: {text[s:e]}")
        #  print("----------------------------------------")
        if entities:
            for ai in range(num_augments):
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

                #  assert text[entity['start']:entity['end'] +
                #              1] == new_entity_text
                text_mention = text[entity['start']:entity['end'] + 1]
                if text_mention != new_entity_text:
                    logger.warning(
                        f"Augment missing: |||{new_entity_text}||| vs |||{text_mention}|||"
                    )

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
                       num_augments=0,
                       allow_overlap=False):
    #  assert seg_backoff >= 0 and seg_backoff <= int(seg_len * 3 / 4)
    assert seg_len >= 0 and seg_backoff >= 0 and seg_backoff <= seg_len
    all_text_entities = []
    labels_map = {}

    logger.warning(f"{len(lines) in lines}")
    num_overlap = 0
    for i, s in enumerate(tqdm(lines)):
        guid = str(i)
        text = s['text']
        entities = s['entities']

        new_entities = []
        used_span = []
        #  logger.debug(f"entities: {entities}")
        entities = sorted(entities, key=lambda e: e.start)
        for entity in entities:
            if entity.category not in ner_labels:
                continue
            s = entity.start
            e = entity.end
            entity.mention = text[s:e + 1]

            overlap = False
            for us in used_span:
                if s > us[0] and s <= us[1] and e > us[1]:
                    overlap = True
                    logger.warning(f"Overlap: ({s},{e}) vs ({us[0],us[1]})")
                    break
                if e >= us[0] and e < us[1] and s < us[0]:
                    overlap = True
                    logger.warning(f"Overlap: ({s},{e}) vs ({us[0],us[1]})")
                    break

                #  if s >= us[0] and s <= us[1]:  # and e > us[1]:
                #      overlap = True
                #      logger.warning(f"Overlap: ({s},{e}) vs ({us[0],us[1]})")
                #      break
                #  if e >= us[0] and e <= us[1]:  # and s < us[0]:
                #      overlap = True
                #      logger.warning(f"Overlap: ({s},{e}) vs ({us[0],us[1]})")
                #      break
                #  if us[0] >= s and us[0] <= e:  # and e > us[1]:
                #      overlap = True
                #      logger.warning(f"Overlap: ({s},{e}) vs ({us[0],us[1]})")
                #      break
                #  if us[1] >= s and us[1] <= e:  # and s < us[0]:
                #      overlap = True
                #      logger.warning(f"Overlap: ({s},{e}) vs ({us[0],us[1]})")
                #      break

                #  if s >= us[0] and s <= us[1]:
                #      overlap = True
                #      logger.warning(f"Overlap: ({s},{e}) vs ({us[0],us[1]})")
                #      break
                #  if e >= us[0] and e <= us[1]:
                #      overlap = True
                #      logger.warning(f"Overlap: ({s},{e}) vs ({us[0],us[1]})")
                #      break
            if overlap:
                num_overlap += 1
                #  if not allow_overlap:
                logger.warning(
                    f"Overlap! {i} mention: {entity.mention}({s}:{e}), used_span: {used_span}"
                )
                continue
            used_span.append((s, e))

            new_entities.append(entity)
        #  logger.warning(f"{len(new_entities)} new_entities")
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

            #      if num_augments > 0:
            #          seg_entities = [
            #              {
            #                  'start': x.start - seg_offset,
            #                  'end': x.end - seg_offset,
            #                  'category': x.category,
            #                  'mention': x.mention
            #              } for x in entities
            #              if x.start >= seg_offset and x.end < seg_end
            #          ]
            #          all_text_entities.append((guid, text_a, seg_entities))
            #
            #          for entity in seg_entities:
            #              label_type = entity['category']
            #              s = entity['start']  # - seg_offset
            #              e = entity['end']  #- seg_offset
            #              #  print(s, e)
            #              assert e >= s
            #              #  logger.debug(
            #              #      f"seg_start: {seg_start}, seg_end: {seg_end}, seg_offset: {seg_offset}"
            #              #  )
            #              #  logger.debug(f"s: {s}, e: {e}")
            #              assert s >= 0 and e < len(seg_text)
            #              #  if s >= len(seg_text) or e >= len(seg_text):
            #              #      continue
            #
            #              entity_text = seg_text[s:e + 1]
            #              #  print(label_type, entity_text)
            #
            #              assert len(entity_text) > 0
            #              if label_type not in labels_map:
            #                  labels_map[label_type] = []
            #              labels_map[label_type].append(entity_text)

            seg_offset += seg_len - seg_backoff

    logger.warning(f"num_overlap: {num_overlap}")

    if num_augments > 0:
        aug_tokens = augment_entities(all_text_entities,
                                      labels_map,
                                      num_augments=num_augments)
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

                #  assert text_a[entity['start']:entity['end'] + 1] == entity['mention']
                pass

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
                              num_augments=0,
                              allow_overlap=False):

    examples = []
    for guid, text_a, text_b, labels, text_offset in data_seg_generator(
            lines,
            ner_labels,
            seg_len=seg_len,
            seg_backoff=seg_backoff,
            num_augments=num_augments,
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
            c = entity['category']
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
    logger.info(f"label_examples: {label_examples}")
    logger.info(f"")
    import numpy as np
    logger.warning(
        f"num train examples: {len(train_lengths)}, num test examples: {len(test_lengths)}"
    )
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


def load_train_val_examples(args,
                            train_data_generator,
                            ner_labels,
                            shuffle=True,
                            train_rate=0.9,
                            num_augments=0,
                            aug_train_only=False):
    logger.warning(
        f"shuffle: {shuffle}, train_rate: {train_rate:.2f}, num_augments: {num_augments}"
    )
    #  from theta.modeling import LabeledText, load_ner_labeled_examples
    lines = []
    #  for guid, text, _, entities in train_data_generator(args.train_file):
    #      sl = LabeledText(guid, text, entities)
    #      lines.append({'guid': guid, 'text': text, 'entities': entities})

    for guid, text, _, tags in train_data_generator(args.train_file):
        sl = LabeledText(guid, text)
        for tag in tags:
            c = tag['category']
            s = tag['start']
            m = tag['mention']
            if len(m) == 0:
                continue
            sl.add_entity(c, s, s + len(m) - 1)
        lines.append({'guid': guid, 'text': text, 'entities': sl.entities})
        if args.max_train_examples > 0 and len(
                lines) >= args.max_train_examples:
            break

    allow_overlap = args.allow_overlap
    if num_augments > 0:
        allow_overlap = False

    logger.warning(f"len(lines): {len(lines)}")

    train_base_examples = load_ner_labeled_examples(
        lines,
        ner_labels,
        seg_len=args.seg_len,
        seg_backoff=args.seg_backoff,
        num_augments=num_augments,
        allow_overlap=allow_overlap)

    if args.train_sample_rate < 1.0:
        num_samples = int(len(train_base_examples) * args.train_sample_rate)
        train_base_examples = train_base_examples[:num_samples]

    logger.warning(f"len(train_base_examples): {len(train_base_examples)}")

    from ..utils import split_train_eval_examples
    train_examples, val_examples = split_train_eval_examples(
        train_base_examples,
        train_rate=train_rate,
        fold=args.fold,
        shuffle=shuffle)

    logger.info(f"Loaded {len(train_examples)} train examples, "
                f"{len(val_examples)} val examples.")
    return train_examples, val_examples


#  def load_train_val_examples(args,
#                              train_data_generator,
#                              ner_labels,
#                              shuffle=True,
#                              train_rate=0.9,
#                              num_augments=0,
#                              aug_train_only=False):
#      logger.warning(
#          f"shuffle: {shuffle}, train_rate: {train_rate:.2f}, num_augments: {num_augments}"
#      )
#      #  from theta.modeling import LabeledText, load_ner_labeled_examples
#      lines = []
#      #  for guid, text, _, entities in train_data_generator(args.train_file):
#      #      sl = LabeledText(guid, text, entities)
#      #      lines.append({'guid': guid, 'text': text, 'entities': entities})
#
#      for guid, text, _, tags in train_data_generator(args.train_file):
#          sl = LabeledText(guid, text)
#          for tag in tags:
#              c = tag['category']
#              s = tag['start']
#              m = tag['mention']
#              if len(m) == 0:
#                  continue
#              sl.add_entity(c, s, s + len(m) - 1)
#          lines.append({'guid': guid, 'text': text, 'entities': sl.entities})
#          if args.max_train_examples > 0 and len(
#                  lines) >= args.max_train_examples:
#              break
#
#      allow_overlap = args.allow_overlap
#      if num_augments > 0:
#          allow_overlap = False
#
#      num_lines = len(lines)
#      num_samples = num_lines
#      if args.train_sample_rate < 1.0:
#          num_samples = int(len(train_base_examples) * args.train_sample_rate)
#          lines = lines[:num_samples]
#      logger.warning(
#          f"Train sample rate: {args.train_sample_rate:.2f}, use {num_samples}/{num_lines} examples."
#      )
#
#      from ..utils import split_train_eval_examples
#      if aug_train_only:
#          train_lines, val_lines = split_train_eval_examples(
#              lines, train_rate=train_rate, fold=args.fold, shuffle=shuffle)
#
#          train_examples = load_ner_labeled_examples(
#              train_lines,
#              ner_labels,
#              seg_len=args.seg_len,
#              seg_backoff=args.seg_backoff,
#              num_augments=num_augments,
#              allow_overlap=allow_overlap)
#
#          val_examples = load_ner_labeled_examples(val_lines,
#                                                   ner_labels,
#                                                   seg_len=args.seg_len,
#                                                   seg_backoff=args.seg_backoff,
#                                                   num_augments=0,
#                                                   allow_overlap=allow_overlap)
#      else:
#          train_base_examples = load_ner_labeled_examples(
#              lines,
#              ner_labels,
#              seg_len=args.seg_len,
#              seg_backoff=args.seg_backoff,
#              num_augments=num_augments,
#              allow_overlap=allow_overlap)
#
#          #  if args.train_sample_rate < 1.0:
#          #      num_samples = int(
#          #          len(train_base_examples) * args.train_sample_rate)
#          #      train_base_examples = train_base_examples[:num_samples]
#
#          logger.warning(f"len(train_base_examples): {len(train_base_examples)}")
#
#          train_examples, val_examples = split_train_eval_examples(
#              train_base_examples,
#              train_rate=train_rate,
#              fold=args.fold,
#              shuffle=shuffle)
#
#      logger.info(f"Loaded {len(train_examples)} train examples, "
#                  f"{len(val_examples)} val examples.")
#      return train_examples, val_examples


def load_test_examples(args, test_data_generator):
    #  from theta.modeling import load_ner_examples
    test_base_examples = load_ner_examples(test_data_generator,
                                           args.test_file,
                                           seg_len=args.seg_len,
                                           seg_backoff=args.seg_backoff)

    logger.info(f"Loaded {len(test_base_examples)} test examples.")
    return test_base_examples


def ner_evaluate(dev_file, reviews_file, eval_data_generator):
    dev_data = []
    for guid, text, _, tags in eval_data_generator(dev_file):
        pos = [(x['start'], x['start'] + len(x['mention'])) for x in tags]
        pos = sorted(pos)
        #  logger.info(f"{pos}")
        dev_data.append(pos)

    reviews_data = []
    from theta.modeling import ner_data_generator
    for guid, text, _, tags in ner_data_generator(reviews_file):
        pos = [(x['start'], x['start'] + len(x['mention'])) for x in tags]
        pos = sorted(pos)
        #  logger.info(f"{pos}")
        reviews_data.append(pos)

    total_right = 0
    total_preds = 0
    total_targets = 0
    P = 0.0
    R = 0.0
    F1 = 0.0
    for pred, target in tqdm(zip(reviews_data, dev_data)):
        num_preds = len(pred)
        num_targets = len(target)
        right = set(pred) & set(target)
        num_right = len(right)

        total_right += num_right
        total_preds += num_preds
        total_targets += num_targets

        if num_preds == 0:
            p = 0.0
            r = 0.0
        else:
            p = num_right / len(pred)
            r = num_right / len(target)
        if p == 0.0 and r == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * p * r / (p + r)
        P += p
        R += r
        F1 += f1
    logger.info(
        f"total_right: {total_right}, total_preds: {total_preds}, total_targets: {total_targets}"
    )
    micro_f1 = F1 / len(dev_data)
    micro_p = P / len(dev_data)
    micro_r = R / len(dev_data)

    logger.warning(
        f"Micro: P: {micro_p:.6f}, R: {micro_r:.6f}, F1: {micro_f1:.6f}")

    macro_p = total_right / total_preds
    macro_r = total_right / total_targets
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r)
    logger.warning(
        f"Macro: P: {macro_p:.6f}, R: {macro_r:.6f}, F1: {macro_f1:.6f}")

    eval_results = {
        'macro_p': macro_p,
        'macro_r': macro_r,
        'macro_f1': macro_f1,
        'micro_p': micro_p,
        'micro_r': micro_r,
        'micro_f1': micro_f1,
    }
    return eval_results
