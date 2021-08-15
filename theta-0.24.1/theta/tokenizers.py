#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

    tokenizer = HFTokenizer(vocab_file)
    text = "中国的英文是China，成立于1949年。"
    encodes = tokenizer.encode(text)

    tokens        : ['[CLS]', '中', '国', '的', '英', '文', '是', 'china', '，', '成', '立', '于', '1949', '年', '。', '[SEP]']
    offsets       : [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 19), (19, 20), (20, 21), (0, 0)]
    token2char    : [-1, 0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 19, 20, -1]
    char2token    : [1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 8, 9, 10, 11, 12, 12, 12, 12, 13, 14]
    ids           : [101, 704, 1744, 4638, 5739, 3152, 3221, 8873, 8024, 2768, 4989, 754, 8594, 2399, 511, 102]
    attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    type_ids      : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

"""

from collections import defaultdict

from loguru import logger

from transformers import BertTokenizer


# Hugging Face Tokenizer
class HFTokenizer:
    def __init__(self,
                 vocab_file,
                 lowercase=True,
                 strip_accents=False,
                 clean_text=True,
                 cc=None):
        self.vocab_file = vocab_file
        self.cc = None
        if cc is not None:
            # pip install opencc-python-reimplemented
            import opencc
            self.cc = opencc.OpenCC(cc)
        from tokenizers import BertWordPieceTokenizer
        self._tokenizer = BertWordPieceTokenizer(self.vocab_file,
                                                 lowercase=lowercase,
                                                 strip_accents=strip_accents,
                                                 clean_text=clean_text)

    def encode(self, text, max_seq_length=None, add_special_tokens=True):
        if self.cc is not None:
            text = self.cc.convert(text)

        encodes = self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens)

        tokens = encodes.tokens
        offsets = encodes.offsets
        ids = encodes.ids
        attention_mask = encodes.attention_mask
        type_ids = encodes.type_ids

        #  len(tokens) -
        #  self._tokenizer.num_special_tokens_to_add(is_pair=False))
        token2char = [-1] * (len(tokens))
        char2token = [-1] * len(text)
        for i, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:
                continue
            token2char[i] = start
            for j in range(start, end):
                char2token[j] = i


        return {
            'tokens': tokens,
            'offsets': offsets,
            'token2char': token2char,
            'char2token': char2token,
            'ids': ids,
            'attention_mask': attention_mask,
            'type_ids': type_ids,
        }

    """
    texts = [
        "中国的英文是China，成立于1949年。",
        "美国的英文是American，成立于1776年。",
    ]
    batch_encodes = tokenizer.batch_encode(texts)

    return:

    tokens:  [['[CLS]', '中', '国', '的', '英', '文', '是', 'china', '，', '成', '立', '于', '1949', '年', '。', '[SEP]'], 
              ['[CLS]', '美', '国', '的', '英', '文', '是', 'american', '，', '成', '立', '于', '177', '##6', '年', '。', '[SEP]']
             ]
    offsets: [[(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 19), (19, 20), (20, 21), (0, 0)], 
              [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 21), (21, 22), (22, 23), (23, 24), (0, 0)]
             ]
    token2char: [[0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 19, 20], 
                 [0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 21, 22, 23]
                ]
    char2token: [[1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 8, 9, 10, 11, 12, 12, 12, 12, 13, 14], 
                 [1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 10, 11, 12, 12, 12, 13, 14, 15]
                ]
    ids: [[101, 704, 1744, 4638, 5739, 3152, 3221, 8873, 8024, 2768, 4989, 754, 8594, 2399, 511, 102], 
          [101, 5401, 1744, 4638, 5739, 3152, 3221, 9735, 8024, 2768, 4989, 754, 10132, 8158, 2399, 511, 102]
         ]
    attention_mask: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    type_ids: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    """

    def batch_encode(self, texts, add_special_tokens=True):
        batch_encodes = defaultdict(list)
        for text in texts:
            encodes = self.encode(text, add_special_tokens=add_special_tokens)
            for k, v in encodes.items():
                batch_encodes[k].append(v)
        return batch_encodes

    def save_vocabulary(self, model_path):
        self._tokenizer.save(model_path + "/vocab.txt")


class CNerTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=False, is_english=False):
        #  super().__init__(vocab_file=str(vocab_file),
        #                   do_lower_case=do_lower_case)
        super(CNerTokenizer, self).__init__(vocab_file=vocab_file,
                                            do_lower_case=do_lower_case)
        self.vocab_file = str(vocab_file)
        self.do_lower_case = do_lower_case
        self.is_english = is_english

    def tokenize(self, text):
        if self.is_english:
            text_tokens = []
            words = text.split(' ')
            for w in words:
                word_tokens = super(CNerTokenizer, self).tokenize(w)
            text_tokens.extend(word_tokens)
        else:
            text_tokens = [c for c in text]

        _tokens = []
        for c in text_tokens:
            if self.do_lower_case:
                c = c.lower()

            #  if super(CNerTokenizer,
            #           self).tokenize(f"{c}", add_special_tokens=True)[1:-1]:
            if c in self.vocab:
                _tokens.append(c)
            else:
                _tokens.append('[UNK]')
        return _tokens


def test(args):

    vocab_file = args.vocab_file

    tokenizer = HFTokenizer(vocab_file)

    #  text = "中国的英文是China，成立于1949年。"
    text = "中国宋代景定四年（1263）黎靖德以类编排，于咸淳二年（ 1270 ） 刊为《朱子语类大全）140卷，即今通行本《朱子语类 》"
    print(f"text: {text}")

    encodes = tokenizer.encode(text, add_special_tokens=False)
    for k, v in encodes.items():
        print(f"{k}: {v}")

    tokenizer_b = CNerTokenizer(vocab_file)
    tokens_b = tokenizer_b.tokenize(text)
    logger.warning(f"tokens_b: {tokens_b}")

    #  texts = [
    #      "中国的英文是China，成立于1949年。",
    #      "美国的英文是American，成立于1776年。",
    #  ]
    #
    #  batch_encodes = tokenizer.batch_encode(texts)
    #  for k, v in batch_encodes.items():
    #      print(f"{k}: {v}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--vocab_file",
        default="/opt/share/pretrained/pytorch/bert-base-chinese/vocab.txt",
        help="Vocab file.")
    args = parser.parse_args()

    test(args)
