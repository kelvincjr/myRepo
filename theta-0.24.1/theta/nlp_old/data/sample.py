#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from typing import Callable, List, Tuple, Union

from loguru import logger
from tqdm import tqdm


class BaseSample:
    def __init__(self,
                 data_file: str = None,
                 data_generator: Callable = None,
                 data_list: List = None,
                 labels_list: List = None):
        self.data_list = data_list if data_list is not None else []
        self.labels_list = labels_list if labels_list is not None else []

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        return self.data_list[idx]

    def _new_instance(self, **kwargs):
        raise NotImplementedError

    def shuffle(self, random_state: int = None):
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(self.data_list)

        return self

    def split(self, ratios: Union[float, List, Tuple] = None):
        if type(ratios) == float:
            train_rate = ratios
            eval_rate = 1.0 - train_rate
            test_rate = 0.0
        elif type(ratios) == list or type(ratios) == tuple:
            if len(ratios) == 2:
                train_rate, eval_rate = ratios
                test_rate = 1.0 - train_rate - eval_rate
            elif len(ratios) == 3:
                train_rate, eval_rate, test_rate = ratios
        else:
            raise ValueError(f"ratios: {ratios} must be float or list.")

        num_total_samples = self.__len__()
        num_train_samples = int(num_total_samples * train_rate)
        if test_rate > 0.0:
            num_eval_samples = int(num_total_samples * eval_rate)
            num_test_samples = num_total_samples - num_train_samples - num_eval_samples
        else:
            num_eval_samples = num_total_samples - num_train_samples
            num_test_samples = 0

        train_samples = self._new_instance(
            data_list=self.data_list[:num_train_samples],
            labels_list=self.labels_list)
        eval_samples = self._new_instance(
            data_list=self.data_list[num_train_samples:num_train_samples +
                                     num_eval_samples],
            labels_list=self.labels_list)
        if num_test_samples > 0:
            test_samples = self._new_instance(
                data_list=self.data_list[-num_test_samples:],
                labels_list=self.labels_list)
            return train_samples, eval_samples, test_samples
        else:
            return train_samples, eval_samples


class GlueSample(BaseSample):
    def __init__(self, **kwargs):
        super(GlueSample, self).__init__(**kwargs)

    def _new_instance(self, **kwargs):
        return GlueSample(**kwargs)

    @property
    def num_labels(self):
        if self.labels_list is None:
            return 0
        else:
            return len(self.labels_list)

    @classmethod
    def from_generator(cls, data_generator: Callable, data_source: str = None):
        samples = GlueSample()
        labels_list = []
        for sid, text_a, text_b, label in data_generator(data_source):
            if label not in labels_list:
                labels_list.append(label)
            samples.data_list.append((sid, text_a, text_b, label))

        labels_list = sorted(labels_list)
        logger.info(f"Loaded labels: {labels_list}")
        if not samples.labels_list:
            samples.labels_list = labels_list

        return samples


class EntitySample(BaseSample):
    def __init__(self, **kwargs):
        super(EntitySample, self).__init__(**kwargs)

    def _new_instance(self, **kwargs):
        return EntitySample(**kwargs)

    @classmethod
    def from_generator(cls, data_generator: Callable, data_source: str = None):
        samples = EntitySample()
        categories = []
        for guid, text, _, json_tags in data_generator(data_source):
            tagged_text = TaggedText(guid, text)
            for json_tag in json_tags:
                entity_tag = EntityTag().from_dict(json_tag)
                tagged_text.add_tag(entity_tag)

                category = entity_tag.category
                if category not in categories:
                    categories.append(category)

            samples.data_list.append(tagged_text)

        logger.info(f"Loaded categories: {categories}")
        if not samples.labels_list:
            samples.labels_list = categories

        return samples


class SPOSample(BaseSample):
    def __init__(self, **kwargs):
        super(SPOSample, self).__init__(**kwargs)

    def _new_instance(self, **kwargs):
        return SPOSample(**kwargs)
