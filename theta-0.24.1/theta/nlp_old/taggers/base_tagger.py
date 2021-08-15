#!/usr/bin/env python
# -*- coding: utf-8 -*-


class BaseTagger:
    def __init__(self, tokenizer, label2id):
        self.tokenizer = tokenizer
        self.label2id = label2id

    def encode(self, examples, *kwarg):
        raise NotImplementedError
