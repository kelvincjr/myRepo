# -*- coding: utf-8 -*-

import os, glob
import numpy as np
import pandas as pd

def remove_tail(s, t):
    i0 = s.find(t)
    if i0 >= 0:
        s = s[:i0]
    return s


def remove_head(s, h):
    i0 = s.find(h)
    if i0 >= 0:
        s = s[i0 + len(h):]
    return s



def get_same_rows(pd_first: pd.DataFrame, pd_second: pd.DataFrame) -> list:
    same_rows = []
    for i, row in pd_first.iterrows():
        if row.PREDICTION == pd_second.iloc[i, :].PREDICTION:
            same_rows.append(row.PREDICTION)
        else:
            same_rows.append(np.nan)
    return pd.DataFrame(same_rows, columns=['subject'])


def remove_nan_rows(data: pd.DataFrame,
                    colname: str,
                    nan_token: str = "__nan__") -> (pd.DataFrame, int):
    num_rows = data.shape[0]
    data[colname] = data[colname].fillna(nan_token)
    nan_list = data[data[colname] == nan_token].index.tolist()
    data = data.drop(nan_list)
    num_removed = num_rows - data.shape[0]
    return data, num_removed


def mix_train_data(train_data, test_data, predicted_results, best_results):
    guess_data = test_data
    guess_data.subject = get_same_rows(predicted_results, best_results)
    mixed_train_data = pd.concat([train_data, guess_data])
    mixed_train_data, num_removed = remove_nan_rows(mixed_train_data,
                                                    "subject")
    print(f"{num_removed} nan rows in mixed_train_data have been removed.")
    return mixed_train_data


def ensure_empty_directory(target_dir):
    import os, glob
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    if len(glob.glob(f"target_dir/*")) > 0:
        logger.warning(f"{target_dir} must be empty")
        return False
    return True


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i:i + n_list2] == list2:
            return i
    return -1


def my_longest_common_substring(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    c1 = {}
    c2 = {}
    for c in s1:
        c1[c] = c1.get(c, 0) + 1
    for c in s2:
        c2[c] = c2.get(c, 0) + 1
    s1_total_chars = len(s1)
    s2_total_chars = len(s2)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for c, n1 in c1.items():
        n2 = c2.get(c, 0)
        TP += min(n1, n2)
        TN += n2 - min(n1, n2)
        FP += n1 - min(n1, n2)
    for c, n2 in c2.items():
        if c not in c1:
            TN += n2

    acc = (TP + FN) / s2_total_chars
    recall = (TP + FN) / s1_total_chars
    f1 = 2.0 * acc * recall / (acc + recall)

    return acc, recall, f1


def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, s1_end, s2_end = 0, 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    s1_end = x
                    s2_end = y
            else:
                m[x][y] = 0
    return s1_end, s2_end, longest, s1[s1_end - longest:s1_end]


def merge_sentence(s1, s2):
    s1_words = s1.split(' ')
    s2_words = s2.split(' ')

    s1_end, s2_end, longest, substring = longest_common_substring(
        s1_words, s2_words)

    if len(substring) != 0:
        head = s1_words[:s1_end - longest]
        tail = s2_words[s2_end:]
        new_sent = head + substring + tail
        return ' '.join(new_sent)
    else:
        return ''


#  if __name__ == '__main__':
def test_longest_common_substring():
    #  s1 = 'we are interested in evaluating two closely related variants. One is a long short-term memory (LSTM) unit, and the other is a gated recurrent unit (GRU) proposed more recently by Cho et al. [2014].'
    #  s2 = 'proposed more recently by Cho et al. [2014]. It is well established in the field that the LSTM unit works well on sequence-based tasks with long-term dependencies, but the latter has only recently been introduced and used in the context of machine translation.'
    #
    #  s1_words = s1.split(' ')
    #  s2_words = s2.split(' ')
    s1 = "中国要伟大的"
    s2 = "美国也要伟大才行"
    s1_words = [w for w in s1]
    s2_words = [w for w in s2]

    # pass the words lists into lcs function,
    # so it would compare 2 sentences word by word instead of character by character.
    s1_end, s2_end, longest, substring = longest_common_substring(
        s1_words, s2_words)
    print(f'find common substring in s1[{s1_end - longest}:{s1_end}]')
    print(f'find common substring in s2[{s2_end - longest}:{s2_end}]')
    print(f'common substring = {substring}')

    # if there is a common part in these 2 sentences,
    # maybe you want to merge them into 1 sentence.
    merged = merge_sentence(s1, s2)
    if len(merged) != 0:
        print(f'merged 2 strings into: {merged}')


def test():
    s1 = "中国要伟大的"
    s2 = "美国也要伟大才行"
    acc, recall, f1 = my_longest_common_substring(s1, s2)
    print(f"s1: {s1}")
    print(f"s2: {s2}")
    print(f"f1: {f1:.3f}, acc: {acc:.3f}, recall: {recall:.3f}")


if __name__ == '__main__':
    #  test()
    test_longest_common_substring()
