#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm, trange
from loguru import logger


def get_ner_results(metric):
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}

    title = f"{' '*24}   acc    recall f1    "
    title += " Right/Pred/True "
    title_len = len(title.encode('gbk'))
    logger.info(f"{'=' * title_len}")
    logger.info(title)
    logger.info(f"{'-' * title_len}")

    sorted_entity_info = sorted(entity_info.items(),
                                key=lambda x: x[1]['f1'],
                                reverse=True)
    total_acc = 0
    total_recall = 0
    total_f1 = 0
    for key, metrics in sorted_entity_info:
        if ':' in key:
            category = key.split(':')[1]
        else:
            category = key
        if metric.ignore_categories and category in metric.ignore_categories:
            disp_key = ' -' + key[:16]
        else:
            disp_key = ' ' + key[:16]
        disp_key += ' ' * (24 - len(disp_key.encode('gbk')))
        info = f"{disp_key} | {metrics['acc']:.4f} {metrics['recall']:.4f} {metrics['f1']:.4f}"
        right = metrics['right']
        found = metrics['found']
        origin = metrics['origin']
        info += f" {right}/{found}/{origin}"
        logger.info(info)
        if metric.ignore_categories and category in metric.ignore_categories:
            pass
        else:
            total_acc += metrics['acc']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']
    if metric.ignore_categories:
        num_categories = len(sorted_entity_info) - len(
            metric.ignore_categories)
    else:
        num_categories = len(sorted_entity_info)
    macro_acc = total_acc / num_categories
    macro_recall = total_recall / num_categories
    macro_f1 = total_f1 / num_categories
    logger.info(f"{'-' * title_len}")

    info = f" Micro{' '*18} | {results['acc']:.4f} {results['recall']:.4f} {results['f1']:.4f}"  #" - loss: {results['loss']:.4f}"
    right = results['right']
    found = results['found']
    origin = results['origin']
    info += f" {right}/{found}/{origin}"
    logger.info(info)

    info = f" Macro{' '*18} | {macro_acc:.4f} {macro_recall:.4f} {macro_f1:.4f}"  #" - loss: {results['loss']:.4f}"
    logger.info(info)

    logger.info(f"{'-' * title_len}")

    return results


def get_ner_preds_reviews(preds, test_dataflow, seg_len, seg_backoff):
    reviews = {}
    category_mentions = {}
    for json_d, tagged_text in tqdm(zip(preds, test_dataflow)):
        guid = tagged_text.guid
        text_offset = tagged_text.text_offset
        text = tagged_text.text
        entities = json_d['entities']

        #  logger.debug(f"tagged_text: {tagged_text}")
        #  logger.info(f"preds: {json_d}")

        #  logger.info(
        #      f"guid: {guid}, offset: {text_offset}, entities: {entities}")

        from ...modeling.ner_utils import LabeledText
        labeled_text = LabeledText(guid, text)
        for c, x0, x1 in entities:
            if x0 >= len(text):
                logger.info(f"x0({x0}) >= len(text)({len(text)})")
                logger.warning(
                    f"pos overflow [{guid}]:({c},{x0},{x1}) text: ({len(text)}) {text}"
                )
            if x0 < 0 or x1 < 0 or x0 > x1:
                logger.warning(f"Invalid x0({x0}) and x1({x1})")

            labeled_text.add_entity(c, x0, x1 - 1)
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


def save_ner_preds(args, preds, test_dataflow):
    reviews, category_mentions = get_ner_preds_reviews(preds, test_dataflow,
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
