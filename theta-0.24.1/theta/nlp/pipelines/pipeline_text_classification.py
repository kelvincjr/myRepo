#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union
from tqdm import tqdm
from loguru import logger

from theta.nlp.data.samples import GlueSamples
from theta.nlp.tasks import GlueData, GlueModel, GlueTask


class TextClassificationPipeline:
    def __init__(self, checkpoint_path, num_labels, glue_labels):
        self.checkpoint_path = checkpoint_path
        self.num_labels = num_labels
        self.glue_labels = glue_labels

    def __call__(self,
                 text: Union[str, list],
                 max_length=512,
                 glue_labels=None):

        # -------------------- Data --------------------
        def data_generator():
            if isinstance(text, str):
                yield 0, text, None, None
            elif isinstance(text, list):
                texts = text
                for i, x in enumerate(texts):
                    yield f"{i}", x, None, None
            else:
                raise TypeError(
                    f"text({type(text)}) type must be str or list.")

        test_samples = GlueSamples(labels_list=glue_labels if glue_labels else
                                   [f"{x}" for x in range(self.num_labels)],
                                   data_generator=data_generator)
        data_args = {
            'max_length': max_length,
        }
        task_data = GlueData(data_args,
                             train_samples=None,
                             test_samples=test_samples)

        # -------------------- model --------------------
        task_model = GlueModel(task_args, self.num_labels)
