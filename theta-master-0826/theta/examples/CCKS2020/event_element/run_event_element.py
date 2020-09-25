#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, random, copy, re
from tqdm import tqdm
from loguru import logger
from pathlib import Path
#import mlflow
from collections import Counter, defaultdict

from theta.utils import load_json_file, split_train_eval_examples
from theta.modeling import LabeledText, load_ner_examples, load_ner_labeled_examples, save_ner_preds, show_ner_datainfo

#  if os.environ['NER_TYPE'] == 'span':
#      from theta.modeling.ner_span import load_model, get_args
#  else:
#      from theta.modeling.ner import load_model, get_args

train_base_file = "../../../../competitions/CCKS2020/event_element/data/event_element_train_data_label.txt"
test_base_file = "../../../../competitions/CCKS2020/event_element/data/event_element_dev_data.txt"

event_elements = {
    "破产清算": ["公司名称", "受理法院", "裁定时间", "公告时间", "公司行业"],
    "重大安全事故": ["公司名称", "公告时间", "伤亡人数", "其他影响", "损失金额"],
    "股东增持": ["增持的股东", "增持金额", "增持开始日期"],
    "股东减持": ["减持的股东", "减持金额", "减持开始日期"],
    "股权质押": ["质押方", "接收方", "质押开始日期", "质押结束日期", "质押金额"],
    "股权冻结": ["被冻结股东", "冻结开始日期", "冻结结束日期", "冻结金额"],
    "高层死亡": ["公司名称", "高层人员", "高层职务", "死亡/失联时间", "死亡年龄"],
    "重大资产损失": ["公司名称", "损失金额", "其他损失", "公告时间"],
    "重大对外赔付": ["公司名称", "赔付对象", "赔付金额", "公告时间"]
}

ner_labels = [
    f"{k}_{v}" for k, elements in event_elements.items() for v in elements
]


# -------------------- Data --------------------
def clean_text(text):
    if text:
        text = text.strip()
        text = re.sub('(<br>)', '', text)
        #  text = re.sub('\t', ' ', text)
    return text


def train_data_generator_0(train_file):

    with open(train_file, 'r') as fr:
        lines = fr.readlines()
        for line in tqdm(lines, desc=f"train & eval"):
            d = json.loads(line)
            guid = d['doc_id']
            text = clean_text(d['content'])

            seg_text = text
            seg_labels = []
            for e in d['events']:
                event_type = e['event_type']
                #  if event_type not in ['破产清算']:  # ['股东减持', '股东增持']:
                #      continue
                for k, v in e.items():
                    if not v:
                        continue

                    if k not in ['event_id', 'event_type']:
                        label = '_'.join((event_type, k))

                        #  if label not in ner_labels:
                        #      ner_labels.append(label)

                        i0 = seg_text.find(v)
                        while i0 >= 0:
                            #  if i0 >= 0:
                            if len(v) == 1:
                                #  if labels[i0] == 'O':
                                #      labels[i0] = f"S-{label}"
                                pass
                            else:
                                seg_labels.append((label, i0, i0 + len(v) - 1))
                            #  break
                            i0 = seg_text.find(v, i0 + len(v))

            sl = LabeledText(guid, text)
            for category, start_char, end_char in seg_labels:
                sl.add_entity(category, start_char, end_char)

            yield guid, text, None, sl.entities


# 上一模型推理结果作为训练数据
def train_data_from_last_generator(train_file):

    train_files = [
        './data/event_element_sampling_0713.json',
    ]

    for train_file in train_files:
        tagged_train_json_data = json.load(open(train_file, 'r'))

        all_labels = tagged_train_json_data['labelCategories']
        id2label = {x['id']: x['text'] for x in all_labels}

        all_entities = tagged_train_json_data['labels']

        content = tagged_train_json_data['content']

        #  re_b = '(\\n[-]+ yanbao\\d\\d\\d\\.txt Begin [-]+\\n\\n)'
        #  re_e = '(\\n[-]+ yanbao\\d\\d\\d\\.txt End [-]+\\n\\n)'
        re_b = '(\\n[-]+ [\d]+ Begin [-]+\\n\\n)'
        re_e = '(\\n[-]+ [\d]+ End [-]+\\n\\n)'
        b_list = []
        for x in re.finditer(re_b, content):
            b_list.append((x.start(), x.end()))
        e_list = []
        for x in re.finditer(re_e, content):
            e_list.append((x.start(), x.end()))

        pages = [(x_b[0], x_b[1], x_e[0], x_e[1])
                 for x_b, x_e in zip(b_list, e_list)]

        logger.warning(f"pages: {pages}")

        for i, page in enumerate(pages):
            head_x0, head_x1, tail_x0, tail_x1 = page

            guid = f"{i}"
            text = content[head_x1:tail_x0]
            sl = LabeledText(guid, text)

            for entity in all_entities:
                s = entity['startIndex']
                e = entity['endIndex'] - 1
                assert e >= s
                if s >= head_x1 and e < tail_x0:
                    sl.add_entity(id2label[entity['categoryId']], s - head_x1,
                                  e - head_x1)
            yield guid, text, None, sl.entities


#  train_data_generator = train_data_from_last_generator
train_data_generator = train_data_generator_0


def load_train_val_examples(args):
    lines = []
    for guid, text, _, entities in train_data_generator(args.train_file):
        sl = LabeledText(guid, text, entities)
        lines.append({'guid': guid, 'text': text, 'entities': entities})

    allow_overlap = args.allow_overlap
    if args.num_augements > 0:
        allow_overlap = False

    train_base_examples = load_ner_labeled_examples(
        lines,
        ner_labels,
        seg_len=args.seg_len,
        seg_backoff=args.seg_backoff,
        num_augements=args.num_augements,
        allow_overlap=allow_overlap)

    train_examples, val_examples = split_train_eval_examples(
        train_base_examples,
        train_rate=args.train_rate,
        fold=args.fold,
        shuffle=True)

    logger.info(f"Loaded {len(train_examples)} train examples, "
                f"{len(val_examples)} val examples.")
    return train_examples, val_examples


def load_eval_examples(eval_file):
    lines = []
    for guid, text, _, entities in train_data_generator_0(eval_file):
        sl = LabeledText(guid, text, entities)
        lines.append({'guid': guid, 'text': text, 'entities': entities})

    allow_overlap = args.allow_overlap
    if args.num_augements > 0:
        allow_overlap = False

    train_base_examples = load_ner_labeled_examples(
        lines,
        ner_labels,
        seg_len=args.seg_len,
        seg_backoff=args.seg_backoff,
        num_augements=0,
        allow_overlap=allow_overlap)

    eval_examples = train_base_examples

    logger.info(f"Loaded {len(eval_examples)} eval examples")
    return eval_examples


def test_data_generator(test_file):
    with open(test_file, 'r') as fr:

        lines = fr.readlines()
        for line in tqdm(lines, desc=f"test"):
            d = json.loads(line.strip())
            guid = d['doc_id']
            text = clean_text(d['content'])

            yield guid, text, None, None


def load_test_examples(args):
    test_base_examples = load_ner_examples(test_data_generator,
                                           args.test_file,
                                           seg_len=args.seg_len,
                                           seg_backoff=args.seg_backoff)

    logger.info(f"Loaded {len(test_base_examples)} test examples.")
    return test_base_examples


#  def get_master_event_type(json_entities):
#      event_types = [json_entity['category'] for json_entity in json_entities]
#      c = Counter(event_types)
#      master_event_type = c.most_common(1)[0][0]
#      return master_event_type

# -----


def get_event_role(event):
    for role, value in event.items():
        if role == 'event_type':
            continue
        break
    return role, value


# 获得事件主体字段名
def get_event_subject_field(event_type):
    elements = event_elements[event_type]
    subject_field = elements[0]
    return subject_field


# 获得事件集合的主事件类型
def get_master_event_type(events):
    event_types = [event['event_type'] for event in events]

    #  # 查找第一个（出现最多？）的事件主字段的类型
    #  for event in events:
    #      event_type = event['event_type']
    #      subject_field = get_event_subject_field(event_type)
    #      role, value = get_event_role(event)
    #      if role == subject_field:
    #          return event_type

    # 查找出现最多次数的事件类型为主事件类型
    if event_types:
        c = Counter(event_types)
        master_event_type = c.most_common(1)[0][0]
    else:
        master_event_type = None
    return master_event_type


def keep_events_completeness(events):
    new_events = []

    if events:
        master_event_type = get_master_event_type(events)
        subject_field = get_event_subject_field(master_event_type)

        has_no_subject = False
        subject0 = None
        for event in events:
            if event['event_type'] != master_event_type:
                continue
            if subject_field in event:
                subject0 = event[subject_field]
            else:
                if subject0:
                    event[subject_field] = subject0
                else:
                    has_no_subject = True
            new_events.append(event)

        if has_no_subject and subject0:
            for event in new_events:
                if subject_field not in event:
                    event[subject_field] = subject0
    return new_events


# 尝试将event0合并到event
# 返回合并成功修改后的event，
# 不可合并则返回None
def try_merge_last(event0, event):
    new_event = None
    if event0['event_type'] == event['event_type']:
        # 先检查两个事件相同key的值是否不同
        same_keys = [k for k in event0.keys() if k in event.keys()]
        if sum([event0[k] != event[k] for k in same_keys]) > 0:
            return None
        diff_keys = [k for k in event0.keys() if k not in event.keys()]
        new_event = event
        if diff_keys:
            for k in diff_keys:
                new_event[k] = event0[k]
    return new_event


# 合并事件last
def merge_events_last(events):

    done = False
    final_events = []

    if not events:
        return done, final_events

    master_event_type = get_master_event_type(events)
    if master_event_type not in ["股权质押"]:
        return done, final_events

    for event0 in events:
        found = False
        #  for i, event in enumerate(final_events):
        #  event = final_events[i]
        #  for i in range(len(final_events) - 1, -1, -1):
        if final_events:
            event = final_events[-1]
            new_event = try_merge_last(event0, event)
            if new_event:
                found = True
                final_events[-1] = new_event
                #  final_events[i] = new_event
                #  break
        if not found:
            final_events.append(event0)

    return True, final_events


# 尝试将event0合并到event
# 返回合并成功修改后的event，
# 不可合并则返回None
def try_merge(event0, event):
    new_event = None
    if event0['event_type'] == event['event_type']:
        # 先检查两个事件相同key的值是否不同
        same_keys = [k for k in event0.keys() if k in event.keys()]
        if sum([event0[k] != event[k] for k in same_keys]) > 0:
            return None
        diff_keys = [k for k in event0.keys() if k not in event.keys()]
        new_event = event
        if diff_keys:
            for k in diff_keys:
                new_event[k] = event0[k]
    return new_event


# 合并事件
def merge_events(events):

    final_events = []
    for event0 in events:
        found = False
        for i, event in enumerate(final_events):
            event = final_events[i]
            new_event = try_merge(event0, event)
            if new_event:
                found = True
                final_events[i] = new_event
                break
        if not found:
            final_events.append(event0)
    return final_events


# 假设事件要素按出现顺序展开，出现事件主体（event_elements列表中第一个元素）时，
# 结束上一事件。展开中的事件若只有主体，则抛弃。
# 事件要素重复时出现时，新建事件并延续事件主体。
def merge_events_by_subject(events):

    final_events = []
    event_type0 = None
    subject_field0 = None
    subject_value0 = None
    current_event = {}

    def append_current_event():
        if current_event and subject_value0 is not None:
            #  if subject_value0 is not None:
            #  current_event['event_type'] = event_type0
            #  current_event[subject_field0] = subject_value0
            new_event = {
                'event_type': event_type0,
                subject_field0: subject_value0
            }
            new_event.update(current_event)
            final_events.append(new_event)

    done = False
    if events:
        #  event_types = [event['event_type'] for event in events]
        #  c = Counter(event_types)
        #  master_event_type = c.most_common(1)[0][0]
        master_event_type = get_master_event_type(events)

        #  logger.debug(f"{master_event_type}")
        if master_event_type in ["股东减持"]:
            for event in events:
                event_type = event['event_type']
                #  elements = event_elements[event_type]
                #  subject_field = elements[0]
                subject_field = get_event_subject_field(event_type)
                if event_type != event_type0:
                    # TODO
                    append_current_event()
                    #  if current_event:
                    #      current_event['event_type'] = event_type0
                    #      current_event[subject_field0] = subject_value0
                    #      final_events.append(current_event)
                    event_type0 = event_type
                    current_event = {}
                    subject_field0 = subject_field
                    subject_value0 = None

                #  for role, value in event.items():
                #      if role == 'event_type':
                #          continue
                #      break
                role, value = get_event_role(event)
                if role == subject_field:
                    append_current_event()
                    #  if current_event:
                    #      current_event['event_type'] = event_type0
                    #      current_event[subject_field0] = subject_value0
                    #      final_events.append(current_event)

                    current_event = {}
                    subject_value0 = value

                else:
                    if role not in current_event.keys():
                        current_event[role] = value
                    else:
                        append_current_event()
                        #  if current_event:
                        #      current_event['event_type'] = event_type0
                        #      current_event[subject_field0] = subject_value0
                        #      final_events.append(current_event)
                        current_event = {}
                        current_event[role] = value

            append_current_event()
            #  if current_event:
            #      current_event['event_type'] = event_type0
            #      current_event[subject_field0] = subject_value0
            #      final_events.append(current_event)
            done = True

    return done, final_events


def process_only_once_events(events):
    final_events = []

    done = False
    if events:
        event_types = [event['event_type'] for event in events]
        c = Counter(event_types)
        master_event_type = c.most_common(1)[0][0]

        #  logger.debug(f"{master_event_type}")
        if master_event_type in ["破产清算", "重大安全事故", "高层死亡", "重大资产损失", "重大对外赔付"]:
            elements = event_elements[master_event_type]

            the_only_event = {}
            the_only_event['event_type'] = master_event_type
            for role in elements:
                values = []
                for event in events:
                    if role in event:
                        values.append(event[role])
                if len(values) > 0:
                    c1 = Counter(values)
                    value = c1.most_common(1)[0][0]
                    the_only_event[role] = value
            final_events = [the_only_event]
            done = True
    return done, final_events


def generate_submission(args):
    reviews_file = f"{args.latest_dir}/{args.dataset_name}_reviews_{args.local_id}.json"
    reviews = json.load(open(reviews_file, 'r'))

    submission_file = f"{args.submissions_dir}/{args.dataset_name}_submission_{args.local_id}.json.txt"
    results = []
    for guid, json_data in tqdm(reviews.items(), desc="events"):

        events = []
        #  output_data = {'doc_id': guid, 'events': []}
        json_entities = json_data['entities']

        for json_entity in json_entities:
            category = json_entity['category']
            mention = json_entity['mention']

            tokens = category.split('_')
            assert len(tokens) >= 2
            event_type = tokens[0]
            event_role = "".join(tokens[1:])
            events.append({'event_type': event_type, event_role: mention})

        done, final_events = process_only_once_events(events)
        if not done:
            done, final_events = merge_events_by_subject(events)
            if not done:
                done, final_events = merge_events_last(events)
                if not done:
                    final_events = merge_events(events)

        final_events = keep_events_completeness(final_events)
        result = {'doc_id': guid, 'events': final_events}
        results.append(result)

    with open(submission_file, 'w') as wt:
        for result in tqdm(results, desc="results"):
            line = json.dumps(result, ensure_ascii=False)
            wt.write(f"{line}\n")

    #  result = {'doc_id': guid, 'events': final_events}
    #          master_event_type = get_master_event_type(json_entities)
    #          if master_event_type in [
    #                  "破产清算",
    #                  "重大安全事故",
    #                  "高层死亡",
    #                  "重大资产损失",
    #                  "重大对外赔付",
    #          ]:
    #              process_only_once_events(json_entities)
    #          #  output_data['entities'].append({
    #          #      'label_type':
    #          #      json_entity['category'],
    #          #      'overlap':
    #          #      0,
    #          #      'start_pos':
    #          #      json_entity['start'],
    #          #      'end_pos':
    #          #      json_entity['end'] + 1
    #          #  })
    #          output_data['entities'] = sorted(output_data['entities'],
    #                                           key=lambda x: x['start_pos'])
    #          output_string = json.dumps(output_data, ensure_ascii=False)
    #          wt.write(f"{output_string}\n")
    #
    logger.info(f"Saved {len(reviews)} lines in {submission_file}")

    from theta.modeling import archive_local_model
    archive_local_model(args, submission_file)


def fix_results_file(args):
    """
    只出现两个相同类型的事件，保留较多项事件，并添加较少项事件中新出现的项。
    整    体：    P: 0.71483 -> 0.74525 R: 0.55818 -> 0.54272 F1: 0.62687 -> 0.62806

    破产清算：    P: 0.76000 -> 0.80967 R: 0.84444 -> 0.85079 F1: 0.80000 -> 0.82972

    重大安全事故: P: 0.80816 -> 0.82500 R: 0.81148 -> 0.81148 F1: 0.80982 -> 0.81818
    高层死亡    : P: 0.95224 -> 0.95796 R: 0.97256 -> 0.97256 F1: 0.96229 -> 0.96520


    重大对外赔付: P: 0.59677 -> 0.66071 R: 0.54412 -> 0.54118 F1: 0.56923 -> 0.59677
    重大资产损失: P: 0.65779 -> 0.70661 R: 0.79724 -> 0.78802 F1: 0.72083 -> 0.74510

    股权冻结    : P: 0.74797 -> 0.78042 R: 0.44231 -> 0.42147 F1: 0.55589 -> 0.54735
    股权质押    : P: 0.80612 -> 0.83241 R: 0.48690 -> 0.46687 F1: 0.60711 -> 0.59822
    股东增持    : P: 0.54664 -> 0.56738 R: 0.49412 -> 0.47059 F1: 0.51905 -> 0.51447
    股东减持    : P: 0.54975 -> 0.56267 R: 0.30191 -> 0.27596 F1: 0.38977 -> 0.37030


    """
    results_file = args.results_file

    #  donot_fixes = ['股权冻结', '股权质押', '股东增持', '股东减持']
    donot_fixes = ['股东减持', '股权质押']
    #  donot_fixes = []

    #  with open(dev_base_file, 'r') as f:
    #      dev_lines = f.readlines()

    fixed_results_file = f"fixed_{os.path.basename(results_file)}"
    fw = open(fixed_results_file, 'w')

    num_dup = 0
    with open(results_file, 'r') as f:
        lines = f.readlines()
        for i, z in enumerate(tqdm(lines, desc='Load')):
            z = json.loads(z)
            events = z['events']

            # 只保留超过项目数据大于2的事件
            #  events = [e for e in events if len(e) > 3]

            #  text = dev_lines[i]  #z['content']
            #  logger.info(text)

            #  new_events = []
            #  for e in events:
            #      #  logger.info(f"{e}")
            #      if e['event_type'] == '股东减持':
            #          if '减持' not in text:
            #              continue
            #      elif e['event_type'] == '股东增持':
            #          if '增持' not in text:
            #              continue
            #      new_events.append(e)
            #  #  logger.info(f"events: {events}")
            #  #  logger.info(f"new_events: {new_events}")
            #  events = new_events

            #  logger.debug(f"events: {events}")
            event_types = [e['event_type'] for e in events]
            # 列出出现重复事件的事件
            if len(event_types) != len(set(event_types)):
                e0 = events[0]
                e1 = events[1]
                if len(event_types) == 2 and event_types[0] == event_types[
                        1] and len(e0) != len(e1):
                    num_dup += 1
                    event_type = event_types[0]
                    if event_type not in donot_fixes:
                        print(f"{json.dumps(z, ensure_ascii=False, indent=2)}")
                        if len(e0) > len(e1):
                            e_more = e0
                            e_less = e1
                        else:
                            e_more = e1
                            e_less = e0
                        for k, v in e_less.items():
                            if k not in e_more:
                                e_more[k] = v
                        events = [e_more]
                        print(
                            f"{json.dumps(events, ensure_ascii=False, indent=2)}"
                        )
                        #  z['events'] = [e_more]
                        #  print(f"{json.dumps(z, ensure_ascii=False, indent=2)}")
                        print("")
            z['events'] = events
            fw.write(f"{json.dumps(z, ensure_ascii=False)}\n")

    logger.info(f"Total {num_dup} duplicate examples.")
    fw.close()
    logger.info(f"Fixed results file: {fixed_results_file}")


def get_top_list(json_file, event_types):
    top_list = {}
    with open(json_file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip()
            json_d = json.loads(line)
            events = json_d['events']
            if events:
                master_event_type = get_master_event_type(events)
                if master_event_type in event_types:
                    top_list[json_d['doc_id']] = line
    return top_list


def merge_all_tops(args):

    candidates_0 = [
        ("./submissions/fixed_event_element_submission_e1355ee8bad711eab6df4e0abd8d028e.json.txt",
         ['破产清算', '重大安全事故', '重大资产损失', '股权冻结', '股权质押', '高层死亡']),
        ("./submissions/fixed_event_element_submission_3a66a2a0ba2011eaa6994e0abd8d028e.json.txt",
         ['股东减持', '股东增持', '重大对外赔付']),
    ]
    merged_result_file_0 = "./submissions/merged_results_0.txt"

    candidates = candidates_0
    merged_result_file = merged_result_file_0

    final_lines = []

    final_top_list = {}
    for json_file, event_types in candidates[1:]:
        top_list = get_top_list(json_file, event_types)
        final_top_list.update(top_list)

    with open(candidates[0][0], 'r') as fr:
        lines = fr.readlines()

    with open(merged_result_file, 'w') as wr:
        for line in lines:
            line = line.strip()
            json_d = json.loads(line)
            doc_id = json_d['doc_id']
            if doc_id in final_top_list:
                line = final_top_list[doc_id]
            wr.write(f"{line}\n")

    logger.info(f"Merged result saved to {merged_result_file}")


def sampling_train_data(train_file, max_samples=5):
    from collections import defaultdict
    selected_examples = defaultdict(list)

    all_examples = [
        (guid, text_a, labels)
        for guid, text_a, _, labels in train_data_generator(train_file)
    ]
    random.shuffle(all_examples)
    for guid, text_a, labels in tqdm(all_examples):
        for entity in labels:
            c = entity.category
            if len(selected_examples[c]) < max_samples:
                selected_examples[c].append((guid, text_a, labels))

    all_selected_examples = {}
    for c, examples in selected_examples.items():
        for guid, text_a, labels in examples:
            if guid not in all_selected_examples:
                all_selected_examples[guid] = (guid, text_a, labels)
    all_selected_examples = [(guid, text_a, labels)
                             for _, (guid, text_a,
                                     labels) in all_selected_examples.items()]
    logger.warning(f"Sampling {len(all_selected_examples)} examples.")

    def sampling_data_generator(_):
        for guid, text_a, labels in all_selected_examples:
            yield guid, text_a, None, labels

    from theta.modeling import to_sampling_poplar
    to_sampling_poplar(args,
                       sampling_data_generator,
                       ner_labels=ner_labels,
                       ner_connections=[],
                       start_page=args.start_page,
                       max_pages=args.max_pages)


from theta.modeling import Params, CommonParams, NerParams, NerAppParams, log_global_params

experiment_params = NerAppParams(
    CommonParams(
        dataset_name="event_element",
        experiment_name="ccks2020_event_element",
        train_file="data/event_element_train_data_label.txt",
        eval_file="data/event_element_train_data_label.txt",
        test_file="data/event_element_dev_data.txt",
        learning_rate=2e-5,
        train_max_seq_length=512,
        eval_max_seq_length=512,
        per_gpu_train_batch_size=4,
        per_gpu_eval_batch_size=4,
        per_gpu_predict_batch_size=4,
        seg_len=510,
        seg_backoff=128,
        num_train_epochs=10,
        fold=0,
        num_augements=3,
        enable_kd=False,
        enable_sda=False,
        sda_teachers=2,
        loss_type="CrossEntropyLoss",
        #  loss_type='FocalLoss',
        focalloss_gamma=2.0,
        model_type="bert",
        model_path=
        #  "/opt/share/pretrained/pytorch/hfl/chinese-electra-large-discriminator",
        "/kaggle/working",
        #  "/opt/share/pretrained/pytorch/bert-base-chinese",
        fp16=True,
        best_index="f1",
        random_type="np"),
    NerParams(ner_labels=ner_labels, ner_type='crf', no_crf_loss=False))

experiment_params.debug()


def main(args):
    from theta.utils import init_theta

    def do_eda(args):
        show_ner_datainfo(ner_labels, train_data_generator, args.train_file,
                          test_data_generator, args.test_file)

    def do_submit(args):
        generate_submission(args)

    if args.do_eda:
        do_eda(args)

    elif args.do_submit:
        init_theta(args)
        do_submit(args)

    elif args.fix_results:
        fix_results_file(args)

    elif args.do_merge:
        merge_all_tops(args)

    elif args.sampling_train_data:
        init_theta(args)
        sampling_train_data(args.train_file)

    elif args.to_train_poplar:
        from theta.modeling import to_train_poplar
        to_train_poplar(args,
                        train_data_generator,
                        ner_labels=ner_labels,
                        ner_connections=[],
                        start_page=args.start_page,
                        max_pages=args.max_pages)

    elif args.to_reviews_poplar:
        from theta.modeling import to_reviews_poplar
        to_reviews_poplar(args,
                          ner_labels=ner_labels,
                          ner_connections=[],
                          start_page=args.start_page,
                          max_pages=args.max_pages)
    else:

        # -------------------- Model --------------------
        #  if args.ner_type == 'span':
        #      from theta.modeling.ner_span import NerTrainer
        #  else:
        #      from theta.modeling.ner import NerTrainer

        class AppTrainer(NerTrainer):
            def __init__(self, args, ner_labels):
                super(AppTrainer, self).__init__(args,
                                                 ner_labels,
                                                 build_model=None)

            #  def on_predict_end(self, args, test_dataset):
            #      super(Trainer, self).on_predict_end(args, test_dataset)

        trainer = AppTrainer(args, ner_labels)

        def do_train(args):
            train_examples, val_examples = load_train_val_examples(args)
            trainer.train(args, train_examples, val_examples)

        def do_eval(args):
            args.model_path = args.best_model_path
            eval_examples = load_eval_examples(args.eval_file)
            model = load_model(args)
            trainer.evaluate(args, model, eval_examples)

        def do_predict(args):
            args.model_path = args.best_model_path
            test_examples = load_test_examples(args)
            model = load_model(args)
            trainer.predict(args, model, test_examples)
            reviews_file, category_mentions_file = save_ner_preds(
                args, trainer.pred_results, test_examples)
            return reviews_file, category_mentions_file

        if args.do_train:
            do_train(args)

        elif args.do_eval:
            do_eval(args)

        elif args.do_predict:
            do_predict(args)
        '''
        elif args.do_experiment:
            if args.tracking_uri:
                mlflow.set_tracking_uri(args.tracking_uri)
            mlflow.set_experiment(args.experiment_name)

            with mlflow.start_run(run_name=f"{args.local_id}") as mlrun:
                log_global_params(args, experiment_params)

                # ----- Train -----
                do_train(args)

                # ----- Predict -----
                do_predict(args)

                # ----- Submit -----
                do_submit(args)
        '''

if __name__ == '__main__':

    def add_special_args(parser):
        parser.add_argument("--sampling_train_data", action="store_true")
        parser.add_argument("--to_train_poplar", action="store_true")
        parser.add_argument("--to_reviews_poplar", action="store_true")
        parser.add_argument("--start_page", type=int, default=0)
        parser.add_argument("--max_pages", type=int, default=100)
        parser.add_argument("--fix_results", action="store_true")
        parser.add_argument("--results_file", type=str, default=None)
        parser.add_argument("--do_merge", action="store_true")
        return parser

    if experiment_params.ner_params.ner_type == 'span':
        from theta.modeling.ner_span import load_model, get_args, NerTrainer
    else:
        from theta.modeling.ner import load_model, get_args, NerTrainer

    args = get_args(experiment_params=experiment_params,
                    special_args=[add_special_args])
    logger.info(f"args: {args}")

    main(args)
