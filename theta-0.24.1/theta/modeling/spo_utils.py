#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import json
import random
from dataclasses import dataclass, field

from loguru import logger
from tqdm import tqdm

from theta.utils import seg_generator

from ..utils import seg_generator

#  import mlflow


class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text, spo_list, text_offset=0):
        self.guid = guid
        self.text = text
        self.spo_list = spo_list
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


def load_spo_examples(data_generator, examples_file, seg_len=0, seg_backoff=0):

    examples = []

    for guid, text, text_b, spo_list in data_generator(examples_file):
        assert text is not None
        for (seg_text, ), text_offset in seg_generator((text, ), seg_len,
                                                       seg_backoff):

            examples.append(
                InputExample(guid=guid,
                             text=seg_text,
                             spo_list=spo_list,
                             text_offset=text_offset))
    logger.info(f"Loaded {len(examples)} examples.")

    return examples


def show_spo_datainfo(predicate_labels, train_data_generator, train_file,
                      test_data_generator, test_file):
    train_lengths = [
        len(text) for guid, text, _, label in train_data_generator(train_file)
    ]

    test_lengths = [
        len(text) for guid, text, _, _ in test_data_generator(test_file)
    ]

    logger.info(f"****** predicate_labels ******")
    logger.info(f"{predicate_labels}")
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


def get_spo_preds_reviews(preds, examples, seg_len, seg_backoff):
    reviews = {}
    category_mentions = {}
    for spoes, input_example in tqdm(zip(preds, examples)):
        guid = input_example.guid
        text_offset = input_example.text_offset
        text = input_example.text

        spo_list = []
        for x in spoes:
            (sub_start, sub_mention), predicate, (obj_start, obj_mention) = x
            spo = {
                'subject': (sub_start + text_offset, sub_mention),
                'predicate': predicate,
                'object': (obj_start + text_offset, obj_mention)
            }
            spo_list.append(spo)

        #  logger.info(
        #      f"guid: {guid}, offset: {text_offset}, entities: {entities}")

        if guid not in reviews:
            reviews[guid] = {
                'guid': guid,
                'text': "",
                'annotated_text': "",
                'spo_list': []
            }
        reviews[guid]['text'] += text[:seg_len - seg_backoff]
        #  reviews[guid]['annotated_text'] += annotated_text
        reviews_spo_list = reviews[guid]['spo_list']
        reviews_spo_list.extend(spo_list)

        reviews_spo_list = sorted(
            reviews_spo_list,
            key=lambda x: x['subject'][0] * 10000 + x['object'][0])
        from ..utils import remove_duplicate_entities
        reviews_spo_list = remove_duplicate_entities(reviews_spo_list)

        reviews[guid]['spo_list'] = reviews_spo_list
        #  logger.info(f"guid: {guid}, tags: {reviews[guid]['tags']}")

    return reviews, category_mentions


def save_spo_preds(args, preds, test_examples):
    assert len(test_examples) == len(preds)

    reviews, category_mentions = get_spo_preds_reviews(preds, test_examples,
                                                       args.seg_len,
                                                       args.seg_backoff)
    reviews_file = f"{args.latest_dir}/{args.dataset_name}_reviews_{args.local_id}.json"
    json.dump(reviews, open(reviews_file, 'w'), ensure_ascii=False, indent=2)
    logger.info(f"Reviews file: {reviews_file}, {len(reviews)} examples.")

    logger.info(f"Total {len(preds)} lines saved to {reviews_file}")

    #  ----- Tracking -----
    #  if args.do_experiment:
    #      logger.debug(f"mlflow tracking uri: {mlflow.get_tracking_uri()}")
    #      mlflow.log_param(f"{args.dataset_name}_reviews_file", reviews_file)
    #      mlflow.log_artifact(reviews_file, args.artifact_path)

    return reviews_file


# 输入：train_data_generator -> (guid, text, None, tags)
#       tag: ((sub_start, sub_mention), predicate, (obj_start, obj_mention))
#       predicate: <sub_type>_<predicate_type>_<obj_type>
# 任务：
#       1. text长文本切分(seg_len, seg_backoff)
#       2. 数据增强
#       3. 按行划分为训练集和验证集


def fix_mention_with_blank(start, mention):
    real_mention = mention.strip()
    if real_mention:
        if real_mention == mention:
            return start, mention
        p0 = mention.find(real_mention)
        assert p0 >= 0, f"mention: {mention}, real_mention: {real_mention}"
        logger.warning(
            f"Fix |{start}|{mention}| to |{start+p0}:{real_mention}|")
        start += p0
        mention = real_mention
        return start, mention
    else:
        return -1, None


def load_spo_labeled_examples(data_generator,
                              predicate_labels,
                              seg_len=0,
                              seg_backoff=0,
                              num_augments=0,
                              allow_overlap=False):

    examples = []
    for guid, text, _, tags in data_generator(None):

        for (seg_text, ), seg_offset in seg_generator((text, ), seg_len,
                                                      seg_backoff):
            # 按text_offset过滤tags

            seg_start = seg_offset
            seg_end = seg_offset + min(seg_len, len(seg_text))
            #  logger.info(f"tags: {tags}")
            #  seg_tags = [
            #      ((x[0][0] - seg_start, x[0][1]), x[1], (x[2][0] - seg_start,
            #                                              x[2][1])) for x in tags
            #      if x[0][0] >= seg_start and x[0][0] + len(x[0][1]) < seg_end
            #      and x[2][0] >= seg_start and x[2][0] + len(x[2][1]) < seg_end
            #  ]
            seg_tags = []
            for x in tags:
                #  s_start, s_mention = x[0]
                #  predicate = x[1]
                #  o_start, o_mention = x[2]

                s_category = x['sub']['category']
                s_start = x['sub']['start']
                s_mention = x['sub']['mention']
                predicate = x['predicate']
                o_category = x['obj']['category']
                o_start = x['obj']['start']
                o_mention = x['obj']['mention']

                #  logger.warning(
                #      f"({s_category}, {s_start}, {s_mention}), {predicate}, ({o_category}, {o_start}, {o_mention})"
                #  )
                #  for seg_tag in seg_tags:
                #  (s_start, s_mention), predicate, (o_start, o_mention) = seg_tag

                s_start, s_mention = fix_mention_with_blank(s_start, s_mention)
                if s_start < 0:
                    continue
                o_start, o_mention = fix_mention_with_blank(o_start, o_mention)
                if o_start < 0:
                    continue
                s_end = s_start + len(s_mention)
                o_end = o_start + len(o_mention)

                if s_start >= seg_start and s_end < seg_end and o_start >= seg_start and o_end < seg_end:
                    s_start -= seg_start
                    o_start -= seg_start
                    s_end = s_start + len(s_mention)
                    o_end = o_start + len(o_mention)

                    assert seg_text[
                        s_start:
                        s_end] == s_mention, f"subject tag: |{seg_text[s_start:s_end]}| != mention: |{s_mention}|. seg: ({seg_start},{seg_end}), seg_tag: {((s_start, s_mention), predicate, (o_start, o_mention))}, seg_text: {seg_text}"
                    assert seg_text[
                        o_start:
                        o_end] == o_mention, f"object tag: |{seg_text[o_start:o_end]}| != mention: |{o_mention}|. seg: (){seg_start}:{seg_end}), seg_tag: {((s_start, s_mention), predicate, (o_start, o_mention))}, seg_text: {seg_text}"
                    seg_tags.append(((s_start, s_mention), predicate,
                                     (o_start, o_mention)))
            if seg_tags:
                examples.append(
                    InputExample(guid=guid,
                                 text=seg_text,
                                 spo_list=seg_tags,
                                 text_offset=seg_start))

    logger.info(f"Loaded {len(examples)} examples.")
    return examples


def load_train_val_examples(args,
                            train_data_generator,
                            predicate_labels,
                            shuffle=True,
                            train_rate=0.9,
                            num_augments=0):
    lines = []

    train_base_examples = load_spo_labeled_examples(
        train_data_generator,
        predicate_labels,
        seg_len=args.seg_len,
        seg_backoff=args.seg_backoff,
        num_augments=num_augments,
        allow_overlap=args.allow_overlap)

    if args.train_sample_rate < 1.0:
        num_samples = int(len(train_base_examples) * args.train_sample_rate)
        train_base_examples = train_base_examples[:num_samples]

    from ..utils import split_train_eval_examples
    train_examples, val_examples = split_train_eval_examples(
        train_base_examples,
        train_rate=train_rate,
        fold=args.fold,
        shuffle=shuffle)

    logger.info(f"Loaded {len(train_examples)} train examples, "
                f"{len(val_examples)} val examples.")
    return train_examples, val_examples


def load_test_examples(args, test_data_generator):
    from theta.modeling import load_spo_examples
    test_base_examples = load_spo_examples(test_data_generator,
                                           args.test_file,
                                           seg_len=args.seg_len,
                                           seg_backoff=args.seg_backoff)

    logger.info(f"Loaded {len(test_base_examples)} test examples.")
    return test_base_examples
