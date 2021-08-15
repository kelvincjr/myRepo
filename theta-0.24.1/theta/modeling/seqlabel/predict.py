# -*- coding: utf-8 -*-
import os, re
import pandas as pd
import numpy as np
from multiprocessing import Process
from loguru import logger
from tqdm import tqdm

print(f"pwd: {os.path.abspath('.')}")
import sys
sys.path.append("seqlabel")
#  from tf_utils import enable_gpu_growth
from predict_config import Config
from data import load_train_data
from data import load_test_data, get_test_data_list
from seqlabel_model import BertModel, find_entities_from_indices
from seqlabel_utils import remove_head, remove_tail

sys.path.append(".")
from article import Section

debug_extract_labels = Config.debug_extract_labels


def post_process_base(s):
    s = s.replace("简历来自：BOSS直聘", '').replace("简历来自：BOSS直", '')
    return s


def post_process_PROJRESP(s):
    if len(s) <= 2:
        return ""
    if s[:1] in ['：', ':', '?']:
        s = s[1:]
    if s[:2] in ['至今']:
        s = s[2:]

    #  logger.debug(f"1 - post_process_PROJRESP: {s}")

    #  logger.debug(f"3 - post_process_PROJRESP: {s}")
    s = remove_head(s, "责任描述")

    #  logger.debug(f"6 - post_process_PROJRESP: {s}")
    re_string = "(\d{4}[\.\/\-\~]\d\d?\s*([—-][—-]?|至)\s*(\d{4}[\.\/\-\~]\d\d?|[至]?今)|\d{4}\s*年\s*\d\d?\s*月\s*[—-]\s*(\d{4}\s*年\s*\d\d?月))"

    m = re.search(re_string, s)
    if m:
        s0 = m.start()
        s = s[:s0]

    #  logger.debug(f"9 - post_process_PROJRESP: {s}")
    s = remove_tail(s, "项目描述")

    #  ignore_headers = [
    #      '该项目', '本项目', '该系统', '本系统', '此系统', '熟练使用', '公司刚开始', '虽然', '为三亚政府',
    #      '公司', '在校', '熟练', '火火免'
    #  ]
    #  for x in ignore_headers:
    #      if s[:len(x)] == x:
    #          return ""

    if s[:1] in ['：', ':', '?']:
        s = s[1:]
    #  logger.debug(f"post_process_PROJRESP: {s}")
    end_of_section = Section.end_of_section.replace(' ', '')
    if s[-len(end_of_section):] == end_of_section:
        s = s[:-len(end_of_section)]
    return s


def post_process_PNAME(s):
    if s[:4] in [
            '技术架构', '山地联盟', '信息化条', '京东金融', '银联前置', '软硬法规', '专题博物', 'Assi',
            '海南一禾', '基于国网', '华为PT', '利用二维', '个人电脑', '北京运通', '时代电气', '社会实践',
            '个人参与', '专业技能', '博通科技', '牛办是专'
    ]:
        return ""

    if s[:2] in ['至今']:
        s = s[2:]

    re_string = "(\d{4}[\.\/\-\~]\d\d?\s*([—-][—-]?|至)\s*(\d{4}[\.\/\-\~]\d\d?|[至]?今)|\d{4}\s*年\s*\d\d?\s*月\s*[—-]\s*(\d{4}\s*年\s*\d\d?月|至今))"

    m = re.search(re_string, s)
    if m:
        s0 = m.start()
        s = s[:s0]

    s = remove_tail(s, "，")
    s = remove_tail(s, "：")
    s = remove_tail(s, "、")
    s = remove_tail(s, ":")
    s = remove_tail(s, "|")
    s = remove_tail(s, "（")
    s = remove_tail(s, "()")
    s = remove_tail(s, "1")
    s = remove_tail(s, "一")
    s = remove_tail(s, "所属公司")
    s = remove_tail(s, "所属部门")
    s = remove_tail(s, "项目描述")
    s = remove_tail(s, "责任描述")
    s = remove_tail(s, "项目周期")
    s = remove_tail(s, "项目内容")
    s = remove_tail(s, "项目介绍")
    s = remove_tail(s, "学历本科")
    s = remove_tail(s, "职位")
    s = remove_tail(s, "UI设计师")
    s = remove_tail(s, "项目职责")
    s = remove_tail(s, "开发环境")
    s = remove_tail(s, "开发框架")
    s = remove_tail(s, "开发项目")
    s = remove_tail(s, "研发工程师")
    s = remove_tail(s, "使用软件")
    s = remove_tail(s, "主要技术")
    s = remove_tail(s, "软件开发平台")
    s = remove_tail(s, "责任心")
    s = remove_tail(s, "SSM框架")
    s = remove_tail(s, "独立完成")
    s = remove_tail(s, "运用python语言")
    s = remove_tail(s, "负责")

    return s


def post_process_WORKDUTY(s):
    i0 = s.find("工作内容")
    if i0 >= 0:
        s = s[:i0]
    return s


def post_process_WORKCONTENT(s):
    ignore_lines = ['到岗时间', '精通', 'C语言']
    for w in ignore_lines:
        if s.find(w) >= 0:
            return ""

    ignore_headers = [
        '本人', '为人', '熟练', '性格', '乐观', '开朗', '熟悉', '“负责任', '联系方式', '全、', '态度积极',
        '这段时间', '因为我从小'
    ]
    for x in ignore_headers:
        if s[:len(x)] == x:
            return ""
    s = remove_tail(s, "项目描述")

    end_of_section = Section.end_of_section.replace(' ', '')
    if s[-len(end_of_section):] == end_of_section:
        s = s[:-len(end_of_section)]

    return s


def post_process(s, category):
    s = post_process_base(s)
    if category == 'PROJRESP':
        s = post_process_PROJRESP(s)
    elif category == "WORKCONTENT":
        s = post_process_WORKCONTENT(s)
    elif category == "PNAME":
        s = post_process_PNAME(s)
    elif category == "WORKDUTY":
        s = post_process_WORKDUTY(s)
    #  logger.debug(f"post_process: {s}")
    return s


def do_extract_indices(test_data,
                       top_k=20,
                       s_threshold=0.01,
                       e_threshold=0.01,
                       min_label_len=8,
                       seg_len=510,
                       seg_backoff=400,
                       bert_model=None):
    if bert_model is None or type(bert_model) is str:
        if bert_model is None:
            from seqlabel_config import get_best_model_weights_file_name
            model_file = get_best_model_weights_file_name(selected_fold)
        elif type(bert_model) is str:
            #  model_file = f"./outputs/resumes_output/seqlabel/best_model_fold0.weights"
            model_file = bert_model

        bert_model = BertModel(Config.config_path,
                               Config.checkpoint_path,
                               Config.dict_path,
                               Config.learning_rate,
                               max_seq_length=seg_len)

        bert_model.build_model()

        bert_model.predict_model.load_weights(model_file)

    def _extract_indices(text_in, c_in, text_offset, top_k, s_threshold,
                         e_threshold):
        return bert_model.determine_entity_positions(text_in,
                                                     c_in,
                                                     text_offset,
                                                     top_k=top_k,
                                                     s_threshold=s_threshold,
                                                     e_threshold=e_threshold,
                                                     seg_len=seg_len,
                                                     seg_backoff=seg_backoff)

    def extract_indices_func():
        from functools import partial
        return partial(_extract_indices,
                       top_k=top_k,
                       s_threshold=s_threshold,
                       e_threshold=e_threshold)

    extract_indices = extract_indices_func()

    def output_results(test_data, extract_indices):
        indices = {}

        def add_to_indices(ID, text_in, c_in, data_indices):
            if data_indices:
                if ID not in indices:
                    indices[ID] = []
                indices[ID].append([text_in, c_in, data_indices])

        # 解决最后一项抽取不出来的问题
        #  end_of_section = "--- End of section ---"
        n = 0
        for ID, text_in, c_in, text_offset in tqdm(
                test_data, desc="SeqLabels predicting"):

            #  need_end_of_section = False
            #  if len(test_data) == 1:
            #      need_end_of_section = True
            #  elif n == len(test_data) - 1:
            #      need_end_of_section = True
            #  elif n < len(test_data) - 1:
            #      next_ID = test_data[n + 1][0]
            #      if next_ID != ID:
            #          need_end_of_section = True
            #  if need_end_of_section:
            #      text_in += end_of_section
            #      if len(text_in) > seg_len:
            #          #  logger.info(f"{ID}: {text_in}")
            #          data_indices = extract_indices(text_in[:seg_len], c_in,
            #                                         text_offset)
            #          add_to_indices(ID, text_in[:seg_len], c_in, data_indices)
            #
            #          text_in = text_in[seg_len - seg_backoff:]
            #          text_offset += seg_len - seg_backoff
            #
            #          #  logger.warning(f"{ID}: {text_in}")

            data_indices = extract_indices(text_in, c_in, text_offset)
            #  logger.debug(f"{data_indices}")
            #  logger.debug(f"{text_in}")
            add_to_indices(ID, text_in, c_in, data_indices)
            n += 1

        return indices

    def resort_indices(indices):
        new_indices = {}
        for ID, X in indices.items():
            Y = [(text_in, c_in, (t, idx, prob, text_offset))
                 for text_in, c_in, data_indices in X
                 for t, idx, prob, text_offset in data_indices]
            Y = sorted(Y, key=lambda x: x[2][1] + x[2][3])
            new_indices[ID] = Y
        return new_indices

    indices = output_results(test_data, extract_indices)
    indices = resort_indices(indices)

    return indices


#  def do_extract_labels(test_data,
#                        top_k=20,
#                        s_threshold=0.01,
#                        e_threshold=0.01,
#                        min_label_len=8):
#      bert_model = BertModel(Config.config_path, Config.checkpoint_path,
#                             Config.dict_path, Config.learning_rate)
#
#      bert_model.build_model()
#
#      def _extract_entity(text_in, c_in, top_k, s_threshold, e_threshold,
#                          min_label_len):
#          return bert_model.extract_entities(text_in,
#                                             c_in,
#                                             top_k=top_k,
#                                             s_threshold=s_threshold,
#                                             e_threshold=e_threshold,
#                                             min_label_len=min_label_len)
#
#      def extract_entity_func():
#          from functools import partial
#          return partial(_extract_entity,
#                         top_k=top_k,
#                         s_threshold=s_threshold,
#                         e_threshold=e_threshold,
#                         min_label_len=min_label_len)
#
#      extract_entity = extract_entity_func()
#
#      model_file = get_best_model_weights_file_name(selected_fold)
#      #  model_file = f"./outputs/resumes_output/seqlabel/best_model_fold0.weights"
#      bert_model.predict_model.load_weights(model_file)
#
#      def output_results(test_data, extract_entity):
#          results = {}
#
#          def add_to_results(ID, label_text):
#              if label_text:
#                  if ID not in results:
#                      results[ID] = []
#                  results[ID].append(label_text)
#
#          last_entity_texts = []
#          last_entities = []
#          last_ID = None
#          last_dangling_entity = None
#          #  for d in tqdm(iter(test_data)):
#          for d in tqdm(test_data, desc="SeqLabels predicting"):
#              ID = d[0]
#
#              # 处理悬垂实体
#              if ID != last_ID:
#                  if last_ID is not None:
#                      if last_dangling_entity:
#                          idx0, prob0, entity_text = last_dangling_entity
#                          add_to_results(last_ID, entity_text)
#                          #  s = f"{last_ID}【Diangling({idx0}:{idx0+len(entity_text)}), prob: {prob0:.6f})】,{entity_text}\n"
#                          #  F.write(s)
#
#                  #  F.write(
#                  #      "\n============================================================\n"
#                  #  )
#              entities, dangling_entity = extract_entity(d[1], d[2])
#
#              selected_entities = 0
#              for entity_text, idx0, idx1, prob0, prob1 in entities:
#                  # FIX, 重叠窗口重复抽取的问题
#                  if ID == last_ID and entity_text in last_entity_texts:
#                      #  if ID == last_ID and idx0 < seg_backoff and idx1 < seg_backoff:
#                      #  F.write(
#                      #      f"!!!!!! ({idx0}:{idx1})({prob0:.6f}:{prob1:.6f} --------> skip {entity_text}"
#                      #  )
#                      continue
#
#                  add_to_results(ID, entity_text)
#                  #  s = f"{ID}【({idx0}:{idx1})({prob0:.6f}:{prob1:.6f})】,{entity_text}\n"
#                  #  #  s = f"{ID},{entity_text}\n"
#                  #  F.write(s)
#                  selected_entities += 1
#              #  if selected_entities == 0:
#              #      F.write("........................................\n")
#              #  else:
#              #      F.write("----------------------------------------\n")
#              last_entity_texts = [
#                  entity_text for entity_text, _, _, _, _ in entities
#              ]
#
#              last_ID = ID
#              last_dangling_entity = dangling_entity
#          return results
#
#      results = output_results(test_data, extract_entity)
#      return results


def seg_long_text(text, seg_len, seg_backoff):
    texts = []
    n_chars = len(text)
    if n_chars > 0:
        s = 0
        while s < n_chars:
            seg_text = text[s:s + seg_len]
            texts.append(seg_text)
            if s + seg_len >= n_chars:
                break
            s += seg_len - seg_backoff
    return texts


def extract_labels(resume_texts,
                   category,
                   top_k=20,
                   s_threshold=0.01,
                   e_threshold=0.01,
                   min_label_len=8,
                   seg_len=510,
                   seg_backoff=400,
                   bert_model=None):
    test_data = []
    for ID, text in resume_texts.items():

        content_text = text
        content_text = content_text.replace(' ', '')
        if content_text:
            seg_texts = seg_long_text(content_text, seg_len, seg_backoff)
            for seg_text in seg_texts:
                test_data.append([ID, seg_text, category])
        #  test_data.append((ID, text, category))

    #  labels = do_extract_labels(test_data,
    #                             top_k=top_k,
    #                             s_threshold=s_threshold,
    #                             e_threshold=e_threshold,
    #                             min_label_len=min_label_len)
    results = extract_labels_from_test_data(test_data,
                                            category,
                                            top_k=top_k,
                                            s_threshold=s_threshold,
                                            e_threshold=e_threshold,
                                            min_label_len=min_label_len,
                                            seg_len=seg_len,
                                            seg_backoff=seg_backoff,
                                            bert_model=bert_model)

    #  print(results)
    labels = {ID: [x[0] for x in items] for ID, items in results.items()}
    #  print(labels)
    return labels


def adjust_test_data(test_data, seg_len, seg_backoff):
    tmp_data = {}
    for ID, text_in, c_in in test_data:
        if ID not in tmp_data:
            tmp_data[ID] = []
        tmp_data[ID].append((text_in, c_in))
    new_test_data = []
    for ID, X in tmp_data.items():
        X = [(ID, text_in, c_in, (seg_len - seg_backoff) * i)
             for i, (text_in, c_in) in enumerate(X)]
        new_test_data += X

    return new_test_data


def extract_labels_from_test_data(test_data,
                                  category,
                                  top_k=20,
                                  s_threshold=0.01,
                                  e_threshold=0.01,
                                  min_label_len=8,
                                  seg_len=510,
                                  seg_backoff=400,
                                  bert_model=None):

    test_data = adjust_test_data(test_data, seg_len, seg_backoff)
    indices = do_extract_indices(test_data,
                                 top_k=top_k,
                                 s_threshold=s_threshold,
                                 e_threshold=e_threshold,
                                 seg_len=seg_len,
                                 seg_backoff=seg_backoff,
                                 bert_model=bert_model)

    full_texts = {}
    for ID, text, category, text_offset in test_data:
        if ID not in full_texts:
            full_texts[ID] = []
        full_texts[ID].append(text)

    for ID, texts in full_texts.items():
        full_text = ""
        for text_in in texts:
            full_text += text_in[:seg_len - seg_backoff]
        if texts:
            full_text += texts[-1][seg_len - seg_backoff:]
        #  logger.debug(f"{full_text}")
        full_texts[ID] = full_text

    if debug_extract_labels:
        for ID, data_indices in indices.items():
            logger.info(f"-------- {ID} --------")
            for text_in, c_in, (tag, idx, prob, off) in data_indices:
                logger.debug(
                    f"{tag}, {idx}, {prob:.3f}, {off}, {idx+off} - {full_texts[ID][idx+off:idx+off+10]}"
                )

    # -----------------------------------------------------------------
    results = {}

    def add_to_results(ID, label_text, idx0, idx1, prob0, prob1):
        if label_text:
            if ID not in results:
                results[ID] = []
            results[ID].append((label_text, idx0, idx1, prob0, prob1))

    last_entity_texts = []
    last_entities = []
    last_ID = None
    last_dangling_entity = None
    last_full_text = ""
    #  for ID, [text_in, c_in, data_indices] in indices.items():
    for ID, data_indices in tqdm(indices.items(), desc="Extract entities:"):
        full_text = full_texts.get(ID, "")
        if debug_extract_labels:
            logger.info(f"full_text: {full_text}")
        entities, dangling_entity = find_entities_from_indices(
            data_indices, min_label_len)

        if debug_extract_labels:
            logger.info(
                f"entities: {entities}, dangling_entity: {dangling_entity}")

        # 处理悬垂实体
        if ID != last_ID:
            if last_ID is not None:
                if last_dangling_entity:
                    idx0, prob0, entity_text, c_in = last_dangling_entity

                    idx1 = len(last_full_text)
                    entity_text = last_full_text[idx0:idx1]
                    #  logger.debug(
                    #      f"diangling entity({idx0}:{idx1}): {entity_text}")
                    entity_text = post_process(entity_text, category)
                    if debug_extract_labels:
                        logger.warning(
                            f"add diangling entity({idx0}:{idx1}): {entity_text}"
                        )
                    if len(entity_text) > 2:
                        add_to_results(last_ID, entity_text, idx0, idx1, prob0,
                                       0.0)
                    #  s = f"{last_ID}【Diangling({idx0}:{idx0+len(entity_text)}), prob: {prob0:.6f})】,{entity_text}\n"
                    #  logger.debug(s)
                    #  F.write(s)

            #  logger.debug(f"\n============================================================\n")

        selected_entities = 0
        for idx0, idx1, prob0, prob1, entity_text, c_in in entities:
            # FIX, 重叠窗口重复抽取的问题
            if ID == last_ID and entity_text in last_entity_texts:
                #  s = f"  !!!!!! ({idx0}:{idx1})({prob0:.6f}:{prob1:.6f} --------> skip {entity_text}"
                #  logger.debug(s)
                continue

            entity_text = full_text[idx0:idx1]
            #  logger.debug(f"entity({idx0}:{idx1}): {entity_text}")
            entity_text = post_process(entity_text, category)
            if debug_extract_labels:
                logger.warning(f"add entity({idx0}:{idx1}): {entity_text}")
            if len(entity_text) > 2:
                add_to_results(ID, entity_text, idx0, idx1, prob0, prob1)
            #  s = f"{ID}【({idx0}:{idx1})({prob0:.6f}:{prob1:.6f})】,{entity_text}\n"
            #  logger.debug(s)
            selected_entities += 1
        #  if selected_entities == 0:
        #      logger.debug("........................................\n")
        #      #  F.write("........................................\n")
        #  else:
        #      logger.debug("----------------------------------------\n")
        last_entity_texts = [
            entity_text for _, _, _, _, entity_text, c_in in entities
        ]
        last_full_text = full_text

        last_ID = ID
        last_dangling_entity = dangling_entity

    if last_ID and last_dangling_entity:
        idx0, prob0, entity_text, c_in = last_dangling_entity

        idx1 = len(last_full_text)
        entity_text = last_full_text[idx0:idx1]

        entity_text = post_process(entity_text, category)
        if debug_extract_labels:
            logger.warning(
                f"add diangling entity({idx0}:{idx1}): {entity_text}")
        if len(entity_text) > 2:
            add_to_results(last_ID, entity_text, idx0, idx1, prob0, 0.0)
        #  s = f"{last_ID}【Diangling({idx0}:{idx0+len(entity_text)}), prob: {prob0:.6f})】,{entity_text}\n"
        #  logger.debug(s)
        #  F.write(s)

    return results


#  def predict_one_fold(test_data, selected_fold, overwrite_result_file=False):
#      from seqlabel_config import get_result_file_name
#      output_file = get_result_file_name(selected_fold)
#      if not overwrite_result_file and os.path.isfile(output_file):
#          print(f"{output_file} exist.")
#          return
#
#      labels = extract_labels_from_test_data(test_data,
#                                             category=Config.category,
#                                             top_k=Config.top_k,
#                                             s_threshold=Config.s_threshold,
#                                             e_threshold=Config.e_threshold,
#                                             min_label_len=Config.min_label_len,
#                                             seg_len=Config.seg_len,
#                                             seg_backoff=Config.seg_backoff)
#
#      with open(output_file, 'w') as F:
#          for ID, entities in labels.items():
#              for entity_text, idx0, idx1, prob0, prob1 in entities:
#                  s = f"{ID}【({idx0:04d}:{idx1:04d})({prob0:.6f}:{prob1:.6f})】| {entity_text}\n"
#                  F.write(s)
#              F.write("\n")
#


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold",
                        type=int,
                        default=-0,
                        help="Special fold model used for predicting")
    parser.add_argument("--overwrite_result_file",
                        action="store_true",
                        help="Overwrite the content of the result file")

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()

    selected_fold = args.fold

    np.random.seed(Config.SEED)

    logger.add(f"log/test_seqlabel_{Config.experiment}_{Config.category}.log")

    if not os.path.exists(Config.results_dir):
        os.makedirs(Config.results_dir)

    #  df_test = load_test_data(Config.raw_test_data_file, Config.reverse_text)
    #  test_data = get_test_data_list(df_test)
    #  logger.info(f"selected fold: {selected_fold}")

    #  predict_one_fold(test_data, selected_fold, args.overwrite_result_file)

    #  batch_folds = 1
    #  total_folds = Config.total_folds
    #  folds = [total_folds - x - 1 for x in range(total_folds)]
    #  while len(folds) > 0:
    #      selected_folds = []
    #      for i in range(min(batch_folds, len(folds))):
    #          selected_folds.append(folds.pop())
    #      for selected_fold in selected_folds:
    #
    #          test_data = get_test_data_list(df_test)
    #
    #          logger.info(f"selected fold: {selected_fold}")
    #
    #          #  process = Process(target=predict_one_fold_old,
    #          #                    args=(test_data, selected_fold, classes, True))
    #          process = Process(target=predict_one_fold,
    #                            args=(test_data, selected_fold, classes, True))
    #          process.start()
    #      process.join()
    #      break
