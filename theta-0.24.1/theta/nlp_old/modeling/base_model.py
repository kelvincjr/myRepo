#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
from loguru import logger


class BaseModel:
    def __init__(self):
        pass

    def load(self, model_path):
        raise NotImplementedError

    def save(self, model_path):
        raise NotImplementedError

    def train(self, train_examples, eval_examples):
        raise NotImplementedError

    def evaluate(self, eval_examples):
        raise NotImplementedError

    def predicgt(self, test_examples):
        raise NotImplementedError


class TransformerModel(BaseModel):
    def __init__(self, args, tagger, trainer, extractor):
        super(TransformerModel, self).__init__()
        self.args = args
        self.tagger = tagger
        self.trainer = trainer
        self.extractor = extractor

    def load_model(self, model_path=None):
        if self.trainer is not None:
            return self.trainer.load_model(model_path)
        return None

    def save_model(self, model_path):
        pass

    def train(self, train_examples, eval_examples):
        train_data = self.tagger.encode(train_examples)
        eval_data = self.tagger.encode(eval_examples)

    def evaluate(self, eval_examples):
        raise NotImplementedError

    def predicgt(self, test_examples):
        raise NotImplementedError
