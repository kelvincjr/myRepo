#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from multiprocessing import Pool, Process, Queue

from loguru import logger


def myinstance(process_args, a, b, c):
    i, num_processes = process_args
    logger.debug(f"instance {i}/{num_processes}, {a}, {b}, {c}")


def myinstance1(process_args):
    i, num_processes = process_args
    logger.debug(f"instance {i}/{num_processes}")


def myinstance2(process_args, a):
    i, num_processes = process_args
    logger.debug(f"instance {i}/{num_processes}, {a}")


def split_list_for_multiprocessing(items,
                                   max_folds=os.cpu_count(),
                                   min_items=0):
    """
    将列表items划分成几个子列表，用于多进程列表数据拆分。
    max_folds: 最多划分的份数，缺省值0表示取CPU数目。
    min_items: 每份最少条数，缺省值0表示自动划分。
    返回：拆分后子列表起止序号对的列表。
    """

    assert max_folds != 0 or min_items != 0
    indices = []
    total_items = len(items)
    if total_items == 0:
        return []

    num_folds = max_folds
    #  if num_folds == 0:
    #      return []

    if total_items <= min_items:
        indices = [(0, total_items)]
    else:
        if num_folds == 0:
            num_folds = total_items // min_items
        fold_items = total_items // num_folds
        if fold_items == 0:
            indices = [(0, total_items)]
        else:
            if fold_items < min_items:
                num_folds = total_items // min_items
            fold_items = total_items // num_folds
            indices = [(fold_items * i, fold_items * (i + 1))
                       for i in range(num_folds)]
            indices[-1] = (fold_items * (num_folds - 1), total_items)

    return indices


def start_processes(my_function, args, num_processes=0):
    if num_processes == 0:
        num_processes = os.cpu_count()
    child_processes = []
    return_values = []
    queues = []
    for i in range(num_processes):
        logger.debug(f"Start process {i}/{num_processes}")
        q = Queue()
        queues.append(q)
        #  logger.warning(f"start call Process {i}")
        child_process = Process(target=my_function,
                                args=[[i, num_processes, q]] + list(args))
        #  logger.warning(f"end call Process {i}")
        child_process.start()
        child_processes.append(child_process)
    logger.warning(f"end start Processes. num_queues = {len(queues)}")
    for process_id, child_process in enumerate(child_processes):
        #  logger.debug(f"begin child_process.join() {process_id}")
        q = queues[process_id]
        #  logger.warning(f"q.get() progress_id = {process_id}")
        v = q.get()
        #  logger.info(f"q.get() OK progress_id = {process_id} len(v) = {len(v)}")
        return_values.append(v)
        child_process.join()
        #  logger.debug(f"end child_process.join() {process_id}")
    logger.info(
        f"Exit start_processes. num_processes = {num_processes}), num_queues = {len(queues)}"
    )

    return return_values


def test_split_list_for_multiprocessing_for_multiprocessing():
    assert split_list_for_multiprocessing([1, 2, 3, 4, 5, 6, 7],
                                          max_folds=3) == [(0, 2), (2, 4),
                                                           (4, 7)]
    assert split_list_for_multiprocessing([1, 2, 3, 4, 5, 6],
                                          max_folds=3) == [(0, 2), (2, 4),
                                                           (4, 6)]
    assert split_list_for_multiprocessing([1, 2, 3, 4, 5, 6, 7],
                                          min_items=3) == [(0, 7)]
    assert split_list_for_multiprocessing([1, 2, 3, 4, 5, 6],
                                          min_items=3) == [(0, 6)]
    assert split_list_for_multiprocessing([1, 2, 3, 4, 5, 6, 7]) == [(0, 7)]
    #
    assert split_list_for_multiprocessing([1, 2, 3, 4, 5, 6, 7],
                                          max_folds=3,
                                          min_items=3) == [(0, 3), (3, 7)]
    assert split_list_for_multiprocessing([1, 2, 3, 4, 5, 6],
                                          max_folds=3,
                                          min_items=3) == [(0, 3), (3, 6)]
    assert split_list_for_multiprocessing([1, 2, 3, 4, 5, 6, 7],
                                          max_folds=3) == [(0, 2), (2, 4),
                                                           (4, 7)]


if __name__ == '__main__':
    start_processes(myinstance, ("abc", 123, [1, 5, 7]), num_processes=4)
    start_processes(myinstance1, (), num_processes=2)
    start_processes(myinstance2, ("test", ), num_processes=1)
    logger.info("End.")
    print(split_list_for_multiprocessing([1, 2, 3, 4, 5, 6, 7], max_folds=3))
    print(split_list_for_multiprocessing([1, 2, 3, 4, 5, 6], max_folds=3))
    print(split_list_for_multiprocessing([1, 2, 3, 4, 5, 6, 7], min_items=3))
    print(split_list_for_multiprocessing([1, 2, 3, 4, 5, 6], min_items=3))
    print(split_list_for_multiprocessing([1, 2, 3, 4, 5, 6, 7]))
    print(
        split_list_for_multiprocessing([1, 2, 3, 4, 5, 6, 7],
                                       max_folds=3,
                                       min_items=3))
    print(
        split_list_for_multiprocessing([1, 2, 3, 4, 5, 6],
                                       max_folds=3,
                                       min_items=3))
    print(split_list_for_multiprocessing([1, 2, 3, 4, 5, 6, 7], max_folds=3))

    test_split_list_for_multiprocessing_for_multiprocessing()
