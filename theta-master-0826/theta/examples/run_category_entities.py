#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, random
from collections import Counter
from tqdm import tqdm
from loguru import logger
from pathlib import Path

from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import sys
sys.path.append('../../')

from theta.utils import init_theta, seg_generator
#  from theta.modeling.trainer import get_default_optimizer_parameters
from theta.modeling.ner import init_labels, load_model, InputExample, examples_to_dataset, load_examples_from_bios_file, NerTrainer

ner_labels = []
seg_len = 510
seg_backoff = 32
fold = 2

# -------------------- Data --------------------
train_base_file = "../../../../competitions/CCKS2020/event_element/data/event_element_train_data_label.txt"
test_base_file = "../../../../competitions/CCKS2020/event_element/data/event_element_dev_data.txt"

all_event_elements = {
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

event_elements = all_event_elements

#  merged_results_best_0.74917.txt
#  2020年5月17日 16:00

#  0.749170977583234
#  {
#      '整体':         'P:0.733126, R:0.765934, F1:0.749171',
#
#      '破产清算':     'P:0.937294, R:0.901587, F1:0.919094',
#      '重大安全事故': 'P:0.922131, R:0.922131, F1:0.922131',
#      '股东减持':     'P:0.578629, R:0.784153, F1:0.665893',
#      '股权质押':     'P:0.767188, R:0.756549, F1:0.761831',
#      '股东增持':     'P:0.598967, R:0.682353, F1:0.637947',
#      '股权冻结':     'P:0.728033, R:0.557692, F1:0.631579',
#      '高层死亡':     'P:0.969789, R:0.978659, F1:0.974203',
#      '重大资产损失': 'P:0.860465, R:0.852535, F1:0.856481',
#      '重大对外赔付': 'P:0.705882, R:0.705882, F1:0.705882'
#  }

#  fixed_event_element_submission_roberta_2e-5_512_4_10_old_crf_no_loss_by_ncrfpp.txt
#  2020年5月16日 18:20
#  0.741167023554604
#  {
#      '整体':         'P:0.731572, R:0.751017, F1:0.741167',  # Top
#
#      '破产清算':     'P:0.937294, R:0.901587, F1:0.919094',  # Top
#      '重大安全事故': 'P:0.915323, R:0.930328, F1:0.922764',  # Top
#      '股东减持':     'P:0.579449, R:0.747268, F1:0.652745',
#      '股权质押':     'P:0.735878, R:0.742681, F1:0.739264',
#      '股东增持':     'P:0.598967, R:0.682353, F1:0.637947',  # Top
#      '股权冻结':     'P:0.745011, R:0.538462, F1:0.625116',
#      '高层死亡':     'P:0.957704, R:0.966463, F1:0.962064',
#      '重大资产损失': 'P:0.860465, R:0.852535, F1:0.856481',  # Top
#      '重大对外赔付': 'P:0.754386, R:0.632353, F1:0.688000'
#  }

#  fixed_event_element_submission_roberta_2e-5_512_4_10_old_crf_noloss.txt
#  2020年5月15日 21:02
#  0.726913970007893
#  {
#      '整体':         'P:0.705747,R:0.7493897477624084,F1:0.7269139700078927',
#
#      '破产清算':     'P:0.917219, R:0.879365, F1:0.897893',
#      '重大安全事故': 'P:0.912863, R:0.901639, F1:0.907216',
#      '股东减持':     'P:0.581590, R:0.759563, F1:0.658768',
#      '股权质押':     'P:0.767188, R:0.756549, F1:0.761831',  # Top
#      '股东增持':     'P:0.527301, R:0.662745, F1:0.587315',
#      '股权冻结':     'P:0.632381, R:0.532051, F1:0.577894',
#      '高层死亡':     'P:0.960725, R:0.969512, F1:0.965099',
#      '重大资产损失': 'P:0.869159, R:0.857143, F1:0.863109',
#      '重大对外赔付': 'P:0.692308, R:0.661765, F1:0.676692'
#  }

#  fixed_event_element_submission_merge_last_gqzy.txt
#  2020年5月7日 17:18
#  0.732050431320504
#  {
#      '整体':         'P:0.716736, R:0.748034, F1:0.732050',
#
#      '破产清算':     'P:0.910596, R:0.873016, F1:0.891410',
#      '重大安全事故': 'P:0.894737, R:0.905738, F1:0.900204',
#      '股东减持':     'P:0.563008, R:0.756831, F1:0.645688',
#      '股权质押':     'P:0.703216, R:0.741140, F1:0.721680',
#      '股东增持':     'P:0.606440, R:0.664706, F1:0.634238',
#      '股权冻结':     'P:0.728033, R:0.557692, F1:0.631579',  # Top
#      '高层死亡':     'P:0.972561, R:0.972561, F1:0.972561',
#      '重大资产损失': 'P:0.854369, R:0.811060, F1:0.832151',
#      '重大对外赔付': 'P:0.750000, R:0.661765, F1:0.703125'
#  }

#  fixed_event_element_submission_roberta_2e-5_512_4_10_ncrfpp_no_loss.txt
#  2020年5月15日 20:56
#  0.731642544150843
#  {
#      '整体':         'P:0.716701, R:0.747220, F1:0.731643',
#
#      '破产清算':     'P:0.909385, R:0.892063, F1:0.900641',
#      '重大安全事故': 'P:0.903361, R:0.881148, F1:0.892116',
#      '股东减持':     'P:0.568856, R:0.733607, F1:0.640811',
#      '股权质押':     'P:0.697101, R:0.741140, F1:0.718447',
#      '股东增持':     'P:0.587940, R:0.688235, F1:0.634146',
#      '股权冻结':     'P:0.741304, R:0.546474, F1:0.629151',
#      '高层死亡':     'P:0.969789, R:0.978659, F1:0.974203',  # Top
#      '重大资产损失': 'P:0.869565, R:0.829493, F1:0.849057',
#      '重大对外赔付': 'P:0.705882, R:0.705882, F1:0.705882'   # Top
#  }

#  fixed_event_element_submission_roberta_2e-5_512_4_10_new_crf_noloss.txt
#  2020年5月14日 22:15
#  0.728845130388505
#  fixed_event_element_submission_roberta_2e-5_512_4_10_new_crf_noloss_fix_zdaqsg.txt
#  2020年5月15日 00:34 股东减持 'P:0.578629, R:0.784153, F1:0.665893'
#  0.728459182314556
#  {
#      '整体':         'P:0.715330, R:0.742880, F1:0.728845',
#
#      '破产清算':     'P:0.889968, R:0.873016, F1:0.881410',
#      '重大安全事故': 'P:0.878151, R:0.856557, F1:0.867220',
#      '股东减持':     'P:0.575727, R:0.784153, F1:0.663968',  # Top
#      '股权质押':     'P:0.707281, R:0.733436, F1:0.720121',
#      '股东增持':     'P:0.598182, R:0.645098, F1:0.620755',
#      '股权冻结':     'P:0.737418, R:0.540064, F1:0.623497',
#      '高层死亡':     'P:0.961078, R:0.978659, F1:0.969789',
#      '重大资产损失': 'P:0.829384, R:0.806452, F1:0.817757',
#      '重大对外赔付': 'P:0.716666, R:0.632353, F1:0.671875'
#  }


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


def merge_all_tops():

    candidates = [
        ("fixed_event_element_submission_roberta_2e-5_512_4_10_old_crf_no_loss_by_ncrfpp.txt",
         ['破产清算', '重大安全事故', '股东增持', '重大资产损失']),
        ("fixed_event_element_submission_merge_last_gqzy.txt", ['股权冻结']),
        ("fixed_event_element_submission_roberta_2e-5_512_4_10_old_crf_noloss.txt",
         ['股权质押']),
        ("fixed_event_element_submission_roberta_2e-5_512_4_10_ncrfpp_no_loss.txt",
         ['高层死亡', '重大对外赔付']),
        ("fixed_event_element_submission_roberta_2e-5_512_4_10_new_crf_noloss_fix_zdaqsg.txt",
         ['股东减持']),
    ]

    final_lines = []

    final_top_list = {}
    for json_file, event_types in candidates[1:]:
        top_list = get_top_list(json_file, event_types)
        final_top_list.update(top_list)

    with open(candidates[0][0], 'r') as fr:
        lines = fr.readlines()

    merged_result_file = "merged_results.txt"
    with open(merged_result_file, 'w') as wr:
        for line in lines:
            line = line.strip()
            json_d = json.loads(line)
            doc_id = json_d['doc_id']
            if doc_id in final_top_list:
                line = final_top_list[doc_id]
            wr.write(f"{line}\n")

    logger.info(f"Merged result saved to {merged_result_file}")


def get_result_file(args):
    return f"{args.output_dir}/event_element_result.json"


def get_submission_file(args):
    return f"{args.output_dir}/event_element_submission.json"


#  def seg_generator(iterables, seg_len, seg_backoff=0):
#      if seg_len <= 0:
#          yield iterables
#      else:
#          # 确保iterables列表中每一项的条目数相同
#          assert sum([len(x)
#                      for x in iterables]) == len(iterables[0]) * len(iterables)
#          n_segs = len(iterables[0]) // seg_len
#          for i in range(n_segs + 1):
#              s0 = seg_len * i
#              s1 = seg_len * (i + 1) if i < n_segs - 1 else len(iterables[0])
#              if s0 < s1:
#                  segs = [x[s0:s1] for x in iterables]
#                  yield segs
#      #  raise StopIteration


def load_train_eval_examples(train_base_file):
    label2id = {}
    id2label = {}
    train_base_examples = []
    with open(train_base_file, 'r') as fr:
        lines = fr.readlines()
        for line in tqdm(lines, desc=f"train & eval"):
            d = json.loads(line)
            guid = d['doc_id']
            text = d['content']
            words = [w for w in text]
            labels = ['O'] * len(words)
            for e in d['events']:
                event_type = e['event_type']
                #  if event_type not in ['破产清算']:  # ['股东减持', '股东增持']:
                #      continue
                for k, v in e.items():
                    if not v:
                        continue

                    if k not in ['event_id', 'event_type']:
                        label = '_'.join((event_type, k))
                        if label not in label2id:
                            n = len(label2id) + 1
                            label2id[label] = n
                            id2label[n] = label
                            ner_labels.append(label)
                        n = label2id[label]
                        i0 = text.find(v)
                        while i0 >= 0:
                            #  if i0 >= 0:
                            if len(v) == 1:
                                #  if labels[i0] == 'O':
                                #      labels[i0] = f"S-{label}"
                                pass
                            else:
                                labels[i0] = f"B-{label}"
                                for j0 in range(1, len(v)):
                                    labels[i0 + j0] = f"I-{label}"
                            i0 = text.find(v, i0 + 1)

            for seg_words, seg_labels in seg_generator((words, labels),
                                                       seg_len, seg_backoff):
                train_base_examples.append(
                    InputExample(guid=guid,
                                 text_a=seg_words,
                                 labels=seg_labels))

            #  if seg_len > 0:
            #      n_segs = len(text) // seg_len
            #      for i in range(n_segs + 1):
            #          s0 = seg_len * i
            #          s1 = seg_len * (i + 1) if i < n_segs - 1 else len(text)
            #          if s0 < s1:
            #              seg_text = text[s0:s1]
            #              seg_words = words[s0:s1]
            #              seg_labels = labels[s0:s1]
            #
            #              train_base_examples.append(
            #                  InputExample(guid=guid,
            #                               text_a=seg_words,
            #                               labels=seg_labels))
            #  else:
            #      train_base_examples.append(
            #          InputExample(guid=guid, text_a=words, labels=labels))
    #  train_base_examples = train_base_examples[:100]

    random.shuffle(train_base_examples)
    train_rate = 0.9

    num_eval_examples = int(len(train_base_examples) * (1 - train_rate))
    num_train_samples = len(train_base_examples) - num_eval_examples

    if fold == 0:
        eval_examples = train_base_examples[num_train_samples:]
        train_examples = train_base_examples[:num_train_samples]
    else:
        s = num_eval_examples * (fold - 1)
        e = num_eval_examples * fold
        eval_examples = train_base_examples[s:e]
        train_examples = train_base_examples[:s] + train_base_examples[e:]
    logger.info(
        f"Loaded {len(train_examples)} train examples, {len(eval_examples)} eval examples."
    )
    return train_examples, eval_examples


def load_test_examples(test_base_file):
    with open(test_base_file, 'r') as fr:

        test_examples = []
        lines = fr.readlines()
        for line in tqdm(lines, desc=f"test"):
            d = json.loads(line)
            guid = d['doc_id']
            words = [w for w in d['content']]

            for seg_words, in seg_generator((words, ), seg_len, seg_backoff):
                test_examples.append(
                    InputExample(guid=guid, text_a=seg_words, labels=None))

    logger.info(f"Loaded {len(test_examples)} test examples.")
    return test_examples


train_examples, eval_examples = load_train_eval_examples(train_base_file)
test_examples = load_test_examples(test_base_file)

logger.info(f"ner_labels: {ner_labels}")
# -------------------- Model --------------------

#  def load_model(args):
#      model = load_pretrained_model(args)
#      model.to(args.device)
#      return model
#

#  def build_model(args):
#      # -------- model --------
#      model = load_pretrained_model(args)
#      model.to(args.device)
#
#      # -------- optimizer --------
#      optimizer_parameters = get_default_optimizer_parameters(
#          model, args.weight_decay)
#      optimizer = AdamW(optimizer_parameters,
#                        lr=args.learning_rate,
#                        correct_bias=False)
#
#      # -------- scheduler --------
#      scheduler = get_linear_schedule_with_warmup(
#          optimizer,
#          num_warmup_steps=args.total_steps * args.warmup_rate,
#          num_training_steps=args.total_steps)
#
#      return model, optimizer, scheduler

# -------------------- Trainer --------------------


class Trainer(NerTrainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args, build_model=None)

    #  def build_model(self, args):
    #      return build_model(args)

    #  def on_predict_end(self, args, test_dataset):
    #      super(Trainer, self).on_predict_end(args, test_dataset)
    #
    #      test_pred_crf_file = Path(args.output_dir) / "test_pred_crf.json"
    #      logger.info(f"Write pred_results to {test_pred_crf_file}")
    #      with open(test_pred_crf_file, "w") as fwriter:
    #          for i, json_d in enumerate(self.pred_results):
    #              json_line = json.dumps(json_d, ensure_ascii=False)
    #
    #              if i < len(self.pred_results) - 1:
    #                  json_line += ",\n"
    #              else:
    #                  json_line += "\n"
    #
    #              fwriter.write(json_line)


def save_predict_results(args, pred_results, pred_results_file, test_examples):
    test_results = {}
    for json_d, example in tqdm(zip(pred_results, test_examples)):
        guid = example.guid
        text = ''.join(example.text_a)

        if guid not in test_results:
            test_results[guid] = {
                "guid": guid,
                "content": "",
                "events": [],
                "tagged_text": ""
            }

        s0 = 0
        tagged_text = test_results[guid]['tagged_text']
        text_offset = len(test_results[guid]['content'])
        for entity in json_d['entities']:
            event_type = entity[0]
            s = entity[1]
            e = entity[2] + 1
            entity_text = text[s:e]
            test_results[guid]['events'].append(
                (event_type, entity_text, text_offset + s, text_offset + e))

            tagged_text += f"{text[s0:s]}\n"
            tagged_text += f"【{event_type} | {entity_text}】\n"
            s0 = e

        tagged_text += f"{text[s0:]}\n"
        test_results[guid]['tagged_text'] = tagged_text
        test_results[guid]['content'] += text

    json.dump(test_results,
              open(f"{pred_results_file}", 'w'),
              ensure_ascii=False,
              indent=2)


def export_bios_file(examples, bios_file):
    fw_bios = open(bios_file, 'w')
    for example in tqdm(examples):
        # for test examples
        labels = example.labels
        if labels is None:
            labels = ['O'] * len(example.text_a)

        for w, t in zip(example.text_a, labels):
            fw_bios.write(f"{w} {t}\n")
        fw_bios.write("\n")
    logger.info(f"export {len(examples)} examples to {bios_file}")


def eda(args):
    export_bios_file(train_examples, args.train_file)
    export_bios_file(eval_examples, args.eval_file)
    export_bios_file(test_examples, args.test_file)


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


# 保持事件集合的事件的完整性
# 1. 只保留一个主事件类型
# 2. 补充缺失的事件主体
# 3. 补充公告时间

announcement_events = ['破产清算', '重大安全事故', '重大资产损失', '重大对外赔付']


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


def analysis_report(report):
    guid = report['guid']

    events = []
    for event in report['events']:
        event_label = event[0]
        event_value = event[1]
        s = event[2]
        e = event[3]

        tokens = event_label.split('_')
        assert len(tokens) >= 2
        event_type = tokens[0]
        event_role = "".join(tokens[1:])
        events.append({'event_type': event_type, event_role: event_value})

    # online: 0.68423
    #  final_events = merge_events(events)

    # online: 0.69393
    done, final_events = process_only_once_events(events)
    if not done:
        done, final_events = merge_events_by_subject(events)
        if not done:
            done, final_events = merge_events_last(events)
            if not done:
                final_events = merge_events(events)

    final_events = keep_events_completeness(final_events)

    result = {'doc_id': guid, 'events': final_events}
    return result


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


def generate_submission(args):
    result_file = get_result_file(args)
    logger.info("Load result file {result_file}")
    reports = json.load(open(result_file, 'r'))
    results = []
    for guid, report in tqdm(reports.items()):
        result = analysis_report(report)
        results.append(result)

    submission_file = get_submission_file(args)
    with open(submission_file, 'w') as fw:
        for result in results:
            line = json.dumps(result, ensure_ascii=False)
            fw.write(f"{line}\n")

    logger.info(f"Submission file saved to {submission_file}")


def load_results(result_file):
    all_events = {}
    fr = open(result_file, 'r')
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        events = json.loads(line)
        all_events[events['doc_id']] = events

    return all_events


import glob


def new_unique_events(e0, events):
    final_t_events = []

    if len(events) == 0:
        return [e0]

    found = False
    for e in events:
        if not found:
            new_event = try_merge(e0, e)
            if new_event:
                final_t_events.append(new_event)
                found = True
            else:
                final_t_events.append(e)
        else:
            final_t_events.append(e)
    if not found:
        final_t_events.append(e0)
    return final_t_events


def merge_multi(args):
    top_events = load_results('merged_results_best_0.74917.txt')

    files = glob.glob("./results/fixed*.txt")
    for result_file in files:
        logger.info(f"{result_file}")
        single_events = load_results(result_file)

        for doc_id, events in single_events.items():
            tt_events = top_events[doc_id]

            t_events = tt_events['events']
            s_events = events['events']
            master_s_event_type = get_master_event_type(s_events)

            if len(t_events) == 0:
                final_events = [
                    e for e in s_events
                    if e['event_type'] == master_s_event_type
                ]
            else:
                if master_s_event_type not in ['股东增持', '股东减持', '股权质押', '股权冻结']:
                    continue

                if master_s_event_type == '股东减持' and t_events:
                    continue

                master_t_event_type = get_master_event_type(t_events)
                if t_events:
                    if master_s_event_type != master_t_event_type:
                        continue

                final_events = [e for e in t_events]
                for e in s_events:
                    if e['event_type'] != master_t_event_type:
                        continue
                    final_events = new_unique_events(e, final_events)

            tt_events['events'] = final_events

    merge_multi_file = "merge_multi.json.txt"
    wr = open(merge_multi_file, 'w')
    for doc_id, events in tqdm(top_events.items(), desc="Write"):
        wr.write(json.dumps(events, ensure_ascii=False))
        wr.write("\n")
    logger.info(f"Merge multi saved to {merge_multi_file}")


def main(args):
    init_theta(args)
    init_labels(args, ner_labels)

    if args.do_eda:
        eda(args)
        return

    if args.do_merge:
        merge_all_tops()
        return

    if args.merge_multi:
        merge_multi(args)
        return

    if args.generate_submission:
        generate_submission(args)
        return

    if args.fix_results:
        fix_results_file(args)
        return

    trainer = Trainer(args)
    tokenizer = trainer.tokenizer

    #  tokenizer = load_pretrained_tokenizer(args)

    #  def examples_to_dataset(examples, label2id, tokenizer, max_seq_length):
    #  from functools import partial
    #  do_examples_to_dataset = partial(examples_to_dataset,
    #                                   label2id=args.label2id,
    #                                   tokenizer=tokenizer)

    # --------------- train phase ---------------
    if args.do_train:
        #  train_examples = load_examples_from_bios_file(args.train_file)
        #  eval_examples = load_examples_from_bios_file(args.eval_file)
        #  train_examples, eval_examples = load_train_eval_examples(
        #      train_base_file)

        #  train_dataset = do_examples_to_dataset(
        #      examples=train_examples, max_seq_length=args.train_max_seq_length)
        #  eval_dataset = do_examples_to_dataset(
        #      examples=eval_examples, max_seq_length=args.eval_max_seq_length)

        trainer.train(args, train_examples, eval_examples)

    # --------------- predict phase ---------------
    if args.do_eval:
        #  eval_examples = load_examples_from_bios_file(args.eval_file)
        #  train_examples, eval_examples = load_train_eval_examples(
        #      train_base_file)
        #  eval_dataset = do_examples_to_dataset(
        #      examples=eval_examples, max_seq_length=args.eval_max_seq_length)

        model = load_model(args)
        trainer.evaluate(args, model, eval_examples)

    # --------------- predict phase ---------------
    if args.do_predict:
        #  test_examples = load_examples_from_bios_file(args.test_file)
        #  test_examples = load_test_examples(test_base_file)
        #  test_dataset = do_examples_to_dataset(
        #      examples=test_examples, max_seq_length=args.eval_max_seq_length)

        model = load_model(args)

        #  s0 = 0
        #  while s0 < len(text):
        #      seg_text = text[s0:s0 + seg_len]
        #      out = find_events_in_text(seg_text)
        #      #  out['doc_id'] = l['doc_id']
        #
        #      if out:
        #          event_outputs += out['events']
        #      s0 += seg_len
        #
        #      if seg_len == 0:
        #          break

        trainer.predict(args, model, test_examples)

        save_predict_results(args, trainer.pred_results, get_result_file(args),
                             test_examples)


def add_special_args(parser):
    parser.add_argument("--do_eda", action="store_true", help="")
    parser.add_argument("--generate_submission", action="store_true", help="")
    parser.add_argument("--results_file", type=str, help="Results file name")
    parser.add_argument("--fix_results",
                        action="store_true",
                        help="Fix results.")
    parser.add_argument("--do_merge",
                        action="store_true",
                        help="Merge results.")
    parser.add_argument("--merge_multi",
                        action="store_true",
                        help="Merge multi.")
    return parser


if __name__ == '__main__':
    from theta.modeling.ner.args import get_args
    args = get_args([add_special_args])
    main(args)
