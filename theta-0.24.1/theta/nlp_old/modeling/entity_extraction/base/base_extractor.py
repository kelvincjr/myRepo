#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型输出结果抽取器
"""


class BaseExtractor:
    def __init__(self):
        pass

    def decode(self, **kwarg):
        raise NotImplementedError
