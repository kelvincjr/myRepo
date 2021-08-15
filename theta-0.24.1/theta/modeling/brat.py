#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

from loguru import logger
from tqdm import tqdm


def generate_brat_annotation_conf(ner_labels, ner_connections):
    annotation_conf_text = "[entities]\n"
    for x in ner_labels:
        annotation_conf_text += f"{x}\n"
    annotation_conf_text += "\n"

    annotation_conf_text += "[relations]\n"
    #  for c, from_entity, to_entity in ner_connections:
    #      annotation_conf_text += "f{c} {from_entity},{to_entity}\n"
    annotation_conf_text += "\n"

    annotation_conf_text += "[events]\n"

    annotation_conf_text += "\n"

    annotation_conf_text += "[attributes]\n"

    annotation_conf_text += "\n"

    return annotation_conf_text


def generate_brat_visual_conf(ner_labels):
    visual_conf_text = "[labels]\n"
    for x in ner_labels:
        visual_conf_text += f"{x} | {x}\n"
    visual_conf_text += "\n"

    brat_bgcolors = [
        '#007bff', '#17a2b8', '#28a745', '#fd7e14', '#e83e8c', '#dc3545',
        '#20c997', '#ffc107', '#007bff'
    ]
    #  [
    #      '#ffccaa', '#8fb2ff', '#7fe2ff', '#6fffdf', '#aaaaee', '#ccccee',
    #      'darkgray', '#007000', '#7fa2ff', '#cf9fff', '#9fc2ff', '#e0ff00',
    #      '#18c59a', '#b4c8ff', '#ffff00'
    #  ]

    brat_fdcolors = []

    visual_conf_text += "[drawing]\n"
    visual_conf_text += "SPAN_DEFAULT\tfgColor:black, bgColor:lightgreen, borderColor:darken\n"
    visual_conf_text += "ARC_DEFAULT\tcolor:black, arrowHead:triangle-5\n"
    visual_conf_text += "\n"
    visual_conf_text += "Alias\tdashArray:3-3, arrowHead:none\n"
    visual_conf_text += "Equiv\tdashArray:3-3, arrowHead:none\n"
    visual_conf_text += "\n"
    for i, x in enumerate(ner_labels):
        visual_conf_text += f"{x} bgColor:{brat_bgcolors[i % len(brat_bgcolors)]}\n"
    visual_conf_text += "\n"

    visual_conf_text += "# Attributes\n"
    visual_conf_text += "ATTRIBUTE_DEFAULT\tglyph:*\n"
    visual_conf_text += "Negation\tbox:crossed, glyph:<NONE>, dashArray:<NONE>\n"
    visual_conf_text += "Confidence\tglyph:↑|↔|↓\n"
    visual_conf_text += "Speculation\tdashArray:3-3, glyph:<NONE>\n"

    visual_conf_text += "[options]\n"
    visual_conf_text += "Arcs bundle:none\n"
    visual_conf_text += "Text direction:ltr\n"
    visual_conf_text += "\n"

    return visual_conf_text


def generate_brat_kb_shortcuts_conf(ner_labels):
    kb_shortcuts_conf_text = ""
    for i, x in enumerate(ner_labels):
        kb_shortcuts_conf_text += f"{i}\t{x}\n"
        if i >= 9:
            break
    return kb_shortcuts_conf_text


def save_brat_file(tagged_text_list,
                   brat_data_dir,
                   ner_labels,
                   ner_connections,
                   start_page=0,
                   max_pages=10):

    if len(tagged_text_list) == 0:
        return

    assert max_pages >= 1
    if max_pages == 1:
        brat_data_file = tagged_text_list[start_page].guid
        p0 = brat_data_file.rfind('.')
        if p0 > 0:
            brat_data_file = brat_data_file[:p0]
        brat_data_file = os.path.join(brat_data_dir, brat_data_file)
    else:
        brat_data_file = os.path.join(brat_data_dir,
                                      f"{max_pages}_{start_page}")

    brat_colorset = [
        '#007bff', '#17a2b8', '#28a745', '#fd7e14', '#e83e8c', '#dc3545',
        '#20c997', '#ffc107', '#007bff'
    ]
    label2id = {x: i for i, x in enumerate(ner_labels)}

    #  for _id, _text in enumerate(ner_connections):
    #      connection_categories.append({'id': _id, 'text': _text})

    brat_content = ""
    brat_ann_content = ""
    num_pages = 0
    page_offset = 0

    eid = 0
    for tagged_text in tqdm(tagged_text_list, desc="brat files"):
        if num_pages < start_page:
            num_pages += 1
            continue
        guid = tagged_text.guid
        text = tagged_text.text

        if max_pages <= 1:
            page_head = ""
            page_tail = ""
        else:
            page_head = f"\n-------------------- {guid} Begin --------------------\n\n"
            page_tail = f"\n-------------------- {guid} End --------------------\n\n"
        brat_content += page_head + f"{text}" + page_tail

        tags = sorted(tagged_text.tags, key=lambda x: x.start)
        for entity_tag in tags:
            category = entity_tag.category
            mention = entity_tag.mention
            start_char = entity_tag.start
            end_char = start_char + len(mention)

            start_char = page_offset + len(page_head) + start_char
            end_char = page_offset + len(page_head) + end_char

            brat_ann_line = f"T{eid+1}\t{category} {start_char} {end_char}\t{mention}\n"
            brat_ann_content += brat_ann_line

            eid += 1

        num_pages += 1
        if num_pages - start_page >= max_pages:
            break

        page_offset = len(brat_content)

    logger.info(f"Saved {brat_data_file}")

    with open(f"{brat_data_file}.txt", 'w') as wt:
        wt.write(brat_content)

    with open(f"{brat_data_file}.ann", 'w') as wt:
        wt.write(brat_ann_content)


def export_brat_files(tagged_text_list,
                      ner_labels,
                      ner_connections,
                      brat_data_dir,
                      max_pages=10):
    import os, shutil
    if os.path.exists(brat_data_dir):
        shutil.rmtree(brat_data_dir)
    os.makedirs(brat_data_dir)

    annotation_conf_text = generate_brat_annotation_conf(
        ner_labels, ner_connections)
    visual_conf_text = generate_brat_visual_conf(ner_labels)
    kb_shortcuts_conf_text = generate_brat_kb_shortcuts_conf(ner_labels)

    with open(os.path.join(brat_data_dir, "annotation.conf"), 'w') as wt:
        wt.write(annotation_conf_text)

    with open(os.path.join(brat_data_dir, "visual.conf"), 'w') as wt:
        wt.write(visual_conf_text)

    with open(os.path.join(brat_data_dir, "kb_shortcuts.conf"), 'w') as wt:
        wt.write(kb_shortcuts_conf_text)

    num_pages = len(tagged_text_list)
    for start_page in range(0, num_pages, max_pages):
        save_brat_file(tagged_text_list,
                       brat_data_dir,
                       ner_labels,
                       ner_connections,
                       start_page=start_page,
                       max_pages=max_pages)


def get_brat_schemas(brat_data_dir):
    ner_labels = []
    ner_connections = []

    import glob
    txt_files = glob.glob(os.path.join(brat_data_dir, "*.txt"))
    txt_files = sorted(txt_files)
    ann_files = [x[:-3] + "ann" for x in txt_files]

    for txt_file, ann_file in tqdm(zip(txt_files, ann_files),
                                   desc="Brat schemas"):
        #  logger.info(f"{txt_file}")

        content = open(txt_file, 'r').read()
        ann_content = open(ann_file, 'r').read()

        tags = []
        for ann_line in ann_content.split('\n'):
            ann_line = ann_line.strip()
            if len(ann_line) == 0:
                continue
            #  logger.info(f"ann_line: {ann_line}")
            toks = ann_line.split('\t')
            #  logger.info(f"toks: {toks}")
            tid, label, mention = ann_line.split('\t')
            c, s, e = label.split(' ')
            ner_labels.append(c)

    ner_labels = list(set(ner_labels))
    ner_connections = list(set(ner_connections))

    return ner_labels, ner_connections


def brat_data_generator(brat_data_dir):

    import glob
    txt_files = glob.glob(os.path.join(brat_data_dir, "*.txt"))
    txt_files = sorted(txt_files)
    ann_files = [x[:-3] + "ann" for x in txt_files]

    for txt_file, ann_file in tqdm(zip(txt_files, ann_files),
                                   desc="Load from brat"):
        #  logger.info(f"{txt_file}")
        guid = os.path.basename(txt_file)

        content = open(txt_file, 'r').read()
        ann_content = open(ann_file, 'r').read()

        tags = []
        for ann_line in ann_content.split('\n'):
            ann_line = ann_line.strip()
            if len(ann_line) == 0:
                continue
            #  logger.info(f"ann_line: {ann_line}")
            toks = ann_line.split('\t')
            #  logger.info(f"toks: {toks}")
            tid, label, mention = ann_line.split('\t')
            c, s, e = label.split(' ')
            s = int(s)
            e = int(e)
            tags.append([c, s, mention])

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

        #  logger.warning(f"pages: {pages}")
        if pages:
            for i, page in enumerate(pages):
                guid, head_x0, head_x1, tail_x0, tail_x1 = page

                text = content[head_x1:tail_x0]

                json_tags = []
                for c, s, m in tags:
                    e = s + len(m) - 1
                    assert e >= s
                    if s >= head_x1 and e < tail_x0:

                        json_tags.append({
                            'category': c,
                            'start': s - head_x1,
                            'mention': m,
                        })
                json_tags = sorted(json_tags, key=lambda x: x['start'])
                yield guid, text, None, json_tags
        else:
            text = content
            json_tags = []
            for c, s, m in tags:
                e = s + len(m) - 1
                assert e >= s

                json_tags.append({
                    'category': c,
                    'start': s,
                    'mention': m,
                })
            json_tags = sorted(json_tags, key=lambda x: x['start'])
            yield guid, text, None, json_tags
