#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------
"""
扩增训练样本数据
随机替换训练数据标签，增加训练样本。
"""

import re
import numpy as np


def findall_substring(text: str, s: str) -> list:
    """
    查找text中所有匹配的文本片断s（可以是正则表达式）
    返回列表 [(start_pos, end_pos, substring, [regex_group1, regex_group2, ...])]

    """

    if s[:1] != "(":
        s = f"({s})"

    mobjs = re.finditer(s, text)
    results = [(m.start(), m_end(), m.group(), m.groups()) for m in mobjs]
    return results


def get_label_frame(text: str, label_positions: list) -> list:
    """
    获取文本的标注填充框架
    label_positions: 填充标记的起止位置对列表。
    返回扣除标记后的文本片断列表。
    例子：
        frame = get_label_frame("我去北京上学", [(2,4)])
        assert frame == ["我去", "上学"]
    """
    assert text and label_positions

    frame = []
    i0 = 0
    for s, e in label_positions:
        frame.append(text[i0:s])
        i0 = e
    frame.append(text[i0:])

    assert len(frame) >= 2
    assert len(frame) == len(label_positions) + 1

    return frame


def get_frame_entities_count(frame: list) -> int:
    """
    返回实体填充模板要求的实体个数
    例子：
        frame = get_label_frame("我去北京上学", [(2,4)])
        num_entities = get_frame_entities_count(frame)
        assert num_entities == 1
    """
    if len(frame) < 2:
        return 0
    else:
        return len(frame) - 1


def fill_frame(frame: list, labels: list):
    """
    填充实体标注模板
    模板要求填充实体个数可调用get_frame_entities_count(frame)获得，
    参数labels列表条目个数必须等于模板要求个数。
    例子：
        frame = get_label_frame("我去北京上学", [(2,4)])
        text = fill_frame(frame, ['上海'])
        assert text == "我去上海上学"
    """
    assert len(list) == len(labels) + 1

    text = "".join(frame_text + label
                   for frame_text, label in zip(frame, labels + [""]))
    return text


# 从训练数据集中获得所有标注文本
def get_all_labels(train_data):
    labels = list(set([label for ID, text, category, label in train_data]))
    return labels


# -------- 获取所有扣除标签的样本框架 --------
def get_all_frames(train_data):
    sample_frames = []
    for ID, text, category, label in tqdm(train_data, desc="Frames"):
        assert len(text) > 0 and len(label) > 0

        found_labels = findall_substring(text, label)
        if len(found_labels) > 0:
            s, e, _, _ = found_labels[0]
            frame = get_label_frame(text, [(s, e)])
            sample_frames.append(frame)
    assert len(sample_frames) == len(train_data)

    return sample_frames


# 实现随机替换标注实体，按标注模板生成新的训练样本。
#  labels = get_all_labels(train_data)
#  frames = get_all_frames(train_data)
def generate_fake_train_data(train_data, frames, labels):

    assert len(labels) == len(frames)

    fake_train_data = []
    for frame, (ID, _, category, _) in tqdm(zip(sample_frames, train_data),
                                     desc="Generate fake"):
        np.random.shuffle(labels)
        picked_label = labels[0]

        fake_text = fill_frame(frame, [picked_label])

        fake_train_data.append(
            (f"{ID}_{n}", fake_text, category, picked_labels))

    return fake_train_data


# ----------------------------------------
def augment_train_data(train_data: list, num_augments=1) -> list:
    aug_train_data = []

    if len(train_data) > 0 and num_augments > 0:
        # -------- 获取所有训练标签 --------
        labels = list(set([label for ID, text, category, label in train_data]))
        assert len(labels) > 0

        categories = list(set([c for _, _, c, _ in train_data]))
        assert len(categories) == 1
        category = categories[0]

        # -------- 获取所有扣除标签的样本框架 --------
        sample_frames = []
        for ID, text, category, label in tqdm(train_data, desc="Frames"):
            assert len(text) > 0 and len(label) > 0

            found_labels = findall_substring(text, label)
            if len(found_labels) > 0:
                s, e, _, _ = found_labels[0]
                frame = get_label_frame(text, [(s, e)])
                sample_frames.append(frame)
        assert len(sample_frames) == len(train_data)

        for frame, (ID, _, _, _) in tqdm(zip(sample_frames, train_data),
                                         desc="Augementing"):
            for n in range(num_augments):
                np.random.shuffle(labels)
                picked_labels = [labels[0]]

                assert get_frame_entities_count(frame) == len(picked_labels)

                fake_text = fill_frame(frame, picked_labels)

                aug_train_data.append(
                    (f"{ID}_{n}", fake_text, category, picked_labels[0]))

    return aug_train_data
