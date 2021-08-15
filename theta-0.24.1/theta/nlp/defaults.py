#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .arguments import TaskArguments


def get_default_args():
    task_args = TaskArguments.parse_args()
    return task_args
