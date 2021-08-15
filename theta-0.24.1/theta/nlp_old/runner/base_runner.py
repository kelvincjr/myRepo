#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from abc import ABCMeta, abstractmethod


class BaseRunner(metaclass=ABCMeta):
    """
    - run()
    - train()
    - evaluate()
    - save_checkpoint()
    """
    def __init__(self,
                 model,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 meta=None,
                 max_epochs=None,
                 max_iters=None):
        self.model = model
        self.batch_processor = batch_processor
        self.optimizer = optimizer
        self.meta = meta
        if isinstance(work_dir, str):
            self.work_dir = os.path.abspath(work_dir)
        else:
            self.work_dir = None
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__
        self.hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                "Only one of 'max_epochs' or 'max_iters' can be set.")
        self._max_epochs = max_epochs
        self._max_iters = max_iters

        @property
        def model_nae(self):
            return self._model_name

        @property
        def hooks(self):
            return self._hooks

        @property
        def epoch(self):
            return self._epoch

        @property
        def iter(self):
            return self._iter

        @property
        def inner_iter(self):
            return self._inner_iter

        @property
        def max_epochs(self):
            return self._max_epochs

        @property
        def max_iters(self):
            return self._max_iters

        @abstractmethod
        def train(self):
            pass

        @abstractmethod
        def valuate(self):
            pass

        @abstractmethod
        def run(self, data_loaders, workflow, **kwargs):
            pass

        @abstractmethod
        def save_checkpoint(self,
                            out_dir,
                            filename_tmpl,
                            save_optimizer=False,
                            meta=None,
                            create_symlink=True):
            pass

        def current_lr(self):
            if isinstance(self.optimizer, torch.optim.Optimizer):
                lr = [group['lr'] for group in self.optimizer.param_groups]
            elif isinstance(self.optimizer, dict):
                lr = dict()
                for name, optim in self.optimizer.items():
                    lr[name] = [group['lr'] for group in optim.param_groups]
            else:
                raise RuntimeError(
                    f"Optimizer type {type(self.optimizer)} is not in 'torch.optim.Optimizer' or 'dict'."
                )
            return lr

        def load_checkpoint(self, filename, map_location='cpu', strict=False):
            pass
