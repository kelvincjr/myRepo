#!/usr/bin/env python
# -*- coding: utf-8 -*-

from loguru import logger


def get_ner_results(metric):
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}

    title = f"{' '*32}   acc    recall f1    "
    title_len = len(title.encode('gbk'))
    logger.info(f"{'=' * title_len}")
    logger.info(title)
    logger.info(f"{'-' * title_len}")

    sorted_entity_info = sorted(entity_info.items(),
                                key=lambda x: x[1]['f1'],
                                reverse=True)
    for key, metrics in sorted_entity_info:
        #  for key in entity_info.keys():
        #      metrics = entity_info[key]
        disp_key = key[:16]
        disp_key += ' ' * (32 - len(disp_key.encode('gbk')))
        info = f"{disp_key} | {metrics['acc']:.4f} {metrics['recall']:.4f} {metrics['f1']:.4f}"
        logger.info(info)
    logger.info(f"{'-' * title_len}")
    info = f"{' '*32} | {results['acc']:.4f} {results['recall']:.4f} {results['f1']:.4f}"  #" - loss: {results['loss']:.4f}"
    logger.info(info)
    logger.info(f"{'-' * title_len}")

    return results

