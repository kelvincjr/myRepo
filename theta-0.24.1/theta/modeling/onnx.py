#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from loguru import logger
import onnxruntime

from .trainer import generate_dataloader
from ..utils.progbar import Progbar


def export_onnx(args,
                model,
                dataset,
                onnx_file,
                input_names=None,
                output_names=None):
    inputs = tuple([x.cuda() for x in list(dataset.tensors)[:-1]])
    torch.onnx.export(model,
                      args=inputs,
                      f=onnx_file,
                      input_names=input_names,
                      output_names=output_names)

    logger.info(f"Export onnx model to {onnx_file}.")


def inference_from_onnx(args, onnx_file, test_dataset):
    test_dataloader = generate_dataloader(args, test_dataset,
                                          args.per_gpu_predict_batch_size)
    session = onnxruntime.InferenceSession(onnx_file, None)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    for i, input in enumerate(inputs):
        logger.debug(f"onnx inputs[{i}]: {input}")
    for i, output in enumerate(outputs):
        logger.debug(f"onnx outputs[{i}]: {output}")

    pbar = Progbar(target=len(test_dataloader), desc=f"Predicting")

    logits = []
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels = batch

        onnx_args = {
            inputs[i].name: x.detach().cpu().numpy()
            for i, x in enumerate(batch[:-1])
        }

        batch_logits = [session.run([], onnx_args)[0].tolist()[0]]

        logits += batch_logits
        pbar.update(step + 1)

    logits = np.array(logits)
    logger.debug(f"logits[:10]: {logits[:10]}")

    return logits
