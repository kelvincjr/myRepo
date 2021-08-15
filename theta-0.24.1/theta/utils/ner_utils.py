#!/usr/bin/env python
# -*- coding: utf-8 -*-

from loguru import logger


def get_ner_results(metric):
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}

    title = f"{' '*24}   acc    recall f1    "
    title += " Right/Pred/True "
    title_len = len(title.encode('gbk'))
    logger.info(f"{'=' * title_len}")
    logger.info(title)
    logger.info(f"{'-' * title_len}")

    sorted_entity_info = sorted(entity_info.items(),
                                key=lambda x: x[1]['f1'],
                                reverse=True)
    total_acc = 0
    total_recall = 0
    total_f1 = 0
    for key, metrics in sorted_entity_info:
        if ':' in key:
            category = key.split(':')[1]
        else:
            category = key
        if metric.ignore_categories and category in metric.ignore_categories:
            disp_key = ' -' + key[:16]
        else:
            disp_key = ' ' + key[:16]
        disp_key += ' ' * (24 - len(disp_key.encode('gbk')))
        info = f"{disp_key} | {metrics['acc']:.4f} {metrics['recall']:.4f} {metrics['f1']:.4f}"
        right = metrics['right']
        found = metrics['found']
        origin = metrics['origin']
        info += f" {right}/{found}/{origin}"
        logger.info(info)
        if metric.ignore_categories and category in metric.ignore_categories:
            pass
        else:
            total_acc += metrics['acc']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']
    if metric.ignore_categories:
        num_categories = len(sorted_entity_info) - len(
            metric.ignore_categories)
    else:
        num_categories = len(sorted_entity_info)
    macro_acc = total_acc / num_categories
    macro_recall = total_recall / num_categories
    macro_f1 = total_f1 / num_categories
    logger.info(f"{'-' * title_len}")

    info = f" Micro{' '*18} | {results['acc']:.4f} {results['recall']:.4f} {results['f1']:.4f}"  #" - loss: {results['loss']:.4f}"
    right = results['right']
    found = results['found']
    origin = results['origin']
    info += f" {right}/{found}/{origin}"
    logger.info(info)

    info = f" Macro{' '*18} | {macro_acc:.4f} {macro_recall:.4f} {macro_f1:.4f}"  #" - loss: {results['loss']:.4f}"
    logger.info(info)

    logger.info(f"{'-' * title_len}")

    return results
