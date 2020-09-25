#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
from loguru import logger
STEP = 2

#  events_0 = json.load(
#      open(
#          './outputs/f7b9ad82bb4411ea906ce8611f2e5a0e/medical_event_reviews_f7b9ad82bb4411ea906ce8611f2e5a0e.json',
#          'r'))
#  events_1 = json.load(
#      open(
#          './outputs/5692590cbc5911eaa90ce8611f2e5a0e/medical_event_reviews_5692590cbc5911eaa90ce8611f2e5a0e.json',
#          'r'))

#  STEP = 2
#
#  events_0 = json.load(
#      open(
#          './outputs/f7b9ad82bb4411ea906ce8611f2e5a0e/medical_event_reviews_f7b9ad82bb4411ea906ce8611f2e5a0e.json',
#          'r'))
#  events_1 = json.load(
#      open(
#          './outputs/515f9590b6e011eabe8de8611f2e5a0e/medical_event_reviews_515f9590b6e011eabe8de8611f2e5a0e.json',
#          'r'))

#  STEP = 3
#  events_0 = json.load(open('./submissions/merge_2_results.json', 'r'))
#  events_1 = json.load(
#      open(
#          './outputs/5692590cbc5911eaa90ce8611f2e5a0e/medical_event_reviews_5692590cbc5911eaa90ce8611f2e5a0e.json',
#          'r'))

#  STEP = 4
#  events_0 = json.load(open('./submissions/merge_3_results.json', 'r'))
#  events_1 = json.load(
#      open(
#          './outputs/c5bf0d78bc6a11eaa317e8611f2e5a0e/medical_event_reviews_c5bf0d78bc6a11eaa317e8611f2e5a0e.json',
#          'r'))

# 0.765544
reviews_0 = '2ad2020ec50a11eaa2f7e8611f2e5a0e'
events_0 = json.load(
    open(
        './outputs/2ad2020ec50a11eaa2f7e8611f2e5a0e/medical_event_reviews_2ad2020ec50a11eaa2f7e8611f2e5a0e.json',
        'r'))

# 0.760053
reviews_1 = '3fae0204c4e711eaaaaae8611f2e5a0e'
events_1 = json.load(
    open(
        './outputs/3fae0204c4e711eaaaaae8611f2e5a0e/medical_event_reviews_3fae0204c4e711eaaaaae8611f2e5a0e.json',
        'r'))

for guid, eevts_0 in tqdm(events_0.items()):
    eevts_1 = events_1[guid]
    entities_0 = eevts_0['entities']
    entities_1 = eevts_1['entities']
    new_entities = []
    for e_1 in entities_1:
        found = False
        for e_0 in entities_0:
            s0 = e_0['start']
            e0 = e_0['end']
            s1 = e_1['start']
            e1 = e_1['end']
            if s0 >= s1 and s0 <= e1:
                continue
            if s1 >= s0 and s1 <= e0:
                continue
            if e0 >= s1 and e0 <= e1:
                continue
            if e1 >= s0 and e1 <= e0:
                continue

            if e_0['category'] != e_1['category'] or e_0['mention'] != e_1[
                    'mention']:
                found = True
                break
        if not found:
            new_entities.append(e_1)
    entities_0 += new_entities

merge_results_file = f"./submissions/merge_{reviews_0}_{reviews_1}_results.json"
json.dump(events_0,
          open(merge_results_file, 'w'),
          ensure_ascii=False,
          indent=2)
logger.info(f"Saved {merge_results_file}")


def generate_submission():
    #  submission_file = f"{args.submissions_dir}/{args.dataset_name}_submission_{args.local_id}.xlsx"
    submission_file = f"./submissions/merge_{reviews_0}_{reviews_1}_results.xlsx"
    import xlwt
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet(f"medical_event")

    worksheet.write(0, 0, label='原文')
    worksheet.write(0, 1, label='肿瘤原发部位')
    worksheet.write(0, 2, label='原发病灶大小')
    worksheet.write(0, 3, label='转移部位')

    #  reviews_file = f"{args.latest_dir}/{args.dataset_name}_reviews_{args.local_id}.json"
    reviews_file = merge_results_file
    reviews = json.load(open(reviews_file, 'r'))

    idx = 1
    for guid, json_data in reviews.items():
        text = json_data['text']
        entities = json_data['entities']
        label_entities = {}
        for entity in entities:
            c = entity['category']
            s = entity['start']
            e = entity['end'] + 1
            entity_text = text[s:e]

            if s > len(text) or e > len(text):
                continue
            if len(entity_text) == 0 or len(entity_text) > 16:
                continue
            if ';' in entity_text or '、' in entity_text:
                continue

            if c not in label_entities:
                label_entities[c] = []
            label_entities[c].append(entity_text)

        worksheet.write(idx, 0, label=text)
        if '肿瘤部位' in label_entities:
            worksheet.write(idx, 1, ','.join(label_entities['肿瘤部位']))
        if '病灶大小' in label_entities:
            worksheet.write(idx, 2, ','.join(label_entities['病灶大小']))
        if '转移部位' in label_entities:
            worksheet.write(idx, 3, ','.join(label_entities['转移部位']))

        idx += 1

    workbook.save(submission_file)

    logger.info(f"Saved {submission_file}")


generate_submission()
