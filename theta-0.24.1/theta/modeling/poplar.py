#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
from loguru import tagger


def save_poplar_file(tagged_text_list,
                     poplar_file,
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

    eid = 0
    for tagged_text in tqdm(tagged_text_list):
        if num_pages < start_page:
            num_pages += 1
            continue
        guid = tagged_text.guid
        text = tagged_text.text

        page_head = f"\n-------------------- {guid} Begin --------------------\n\n"
        page_tail = f"\n-------------------- {guid} End --------------------\n\n"
        poplar_content += page_head + f"{text}" + page_tail

        for entity_tag in tagged_text.tags:
            category = entity_tag.category
            mention = entity_tag.mention
            start_char = entity_tag.start
            end_char = start_char + len(mention) - 1

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

    json.dump(poplar_json,
              open(poplar_file, 'w'),
              ensure_ascii=False,
              indent=2)
    logger.info(f"Saved {poplar_file}")


def poplar_data_generator(poplar_file):
    tagged_train_json_data = json.load(open(poplar_file, 'r'))

    all_labels = tagged_train_json_data['labelCategories']
    id2label = {x['id']: x['text'] for x in all_labels}

    all_entities = tagged_train_json_data['labels']

    content = tagged_train_json_data['content']

    #  re_b = '(\\n[-]+ yanbao\\d\\d\\d\\.txt Begin [-]+\\n\\n)'
    #  re_e = '(\\n[-]+ yanbao\\d\\d\\d\\.txt End [-]+\\n\\n)'
    re_b = '(\\n[-]+ (.*?) Begin [-]+\\n\\n)'
    re_e = '(\\n[-]+ (.*?) End [-]+\\n\\n)'

    b_list = []
    for x in re.finditer(re_b, content):
        b_list.append((x.start(), x.end(), x.groups()[1]))
    e_list = []
    for x in re.finditer(re_e, content):
        e_list.append((x.start(), x.end()))

    pages = [(x_b[2], x_b[0], x_b[1], x_e[0], x_e[1])
             for x_b, x_e in zip(b_list, e_list)]

    logger.warning(f"pages: {pages}")

    for i, page in enumerate(pages):
        guid, head_x0, head_x1, tail_x0, tail_x1 = page

        text = content[head_x1:tail_x0]

        json_tags = []
        for entity in all_entities:
            s = entity['startIndex']
            e = entity['endIndex'] - 1
            m = text[s:e + 1]
            assert e >= s
            if s >= head_x1 and e < tail_x0:
                json_tags.append({
                    'category': id2label[entity['categoryId']],
                    'start': s - head_x1,
                    'mention': m,
                })
        yield guid, text, None, json_tags
