#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.nn import BCELoss, CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (AutoConfig, AutoModelForTokenClassification,
                          BertConfig, BertTokenizer, BertTokenizerFast)
from transformers.modeling_bert import BertModel, BertPreTrainedModel
#from transformers.models.bert import BertModel, BertPreTrainedModel

from ...losses import DiceLoss, FocalLoss, LabelSmoothingCrossEntropy
from ...utils.multiprocesses import (barrier_leader_process,
                                     barrier_member_processes,
                                     is_multi_processes)
#  from .utils import CNerTokenizer
from ..models.linears import PoolerEndLogits, PoolerStartLogits
from ..trainer import Trainer, get_default_optimizer_parameters

#  import mlflow

BertLayerNorm = torch.nn.LayerNorm


def get_active_logits(logits, num_labels):
    # Only keep active parts of the loss
    loss_sig = nn.Sigmoid()
    active_logits = logits.view(-1, num_labels * 2)
    active_logits = loss_sig(active_logits)
    active_logits = active_logits**2

    return active_logits


class BertPnForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertPnForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.focalloss_gamma = config.focalloss_gamma
        self.focalloss_alpha = config.focalloss_alpha
        self.diceloss_weight = config.diceloss_weight

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels * 2)

        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        outputs = (logits,
                   )  # add hidden states and attention if they are here

        #  # Only keep active parts of the loss
        #  loss_sig = nn.Sigmoid()
        #  active_logits = logits.view(-1, self.num_labels * 2)
        #  active_logits = loss_sig(active_logits)
        #  active_logits = active_logits**2
        active_logits = get_active_logits(logits, self.num_labels)
        if labels is not None:

            #  active_logits = get_active_logits(logits, self.num_labels)
            #  active_labels = labels.view(-1, self.num_labels * 2).float()
            active_labels = labels.reshape(-1, self.num_labels * 2).float()

            #  logger.warning(f"active_logits.shape: {active_logits.shape}")
            #  logger.warning(f"active_labels.shape: {active_labels.shape}")

            if self.loss_type == 'FocalLoss':
                loss_fct = FocalLoss(gamma=self.focalloss_gamma,
                                     alpha=self.focalloss_alpha)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss_fct = BCELoss(reduction='none')
                loss = loss_fct(active_logits, active_labels)

            batch_size = input_ids.size(0)
            loss = loss.view(batch_size, -1, self.num_labels * 2)
            loss = torch.mean(loss, 2)
            loss = torch.sum(attention_mask * loss) / torch.sum(attention_mask)
            #  outputs = (loss, ) + outputs
            outputs = (loss, ) + (active_logits, )
        else:
            #  outputs = (active_logits, )
            #  outputs = (torch.tensor(0.0).cuda(), ) + outputs
            outputs = (torch.tensor(0.0).cpu(), ) + (active_logits, )

        return outputs


class SpanEntityScore(object):
    def __init__(self, id2label, ignore_categories=None):
        self.id2label = id2label
        self.ignore_categories = ignore_categories
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision *
                                                 recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter(
            [f"{x[0]}:{self.id2label[x[0]]}" for x in self.origins])
        found_counter = Counter(
            [f"{x[0]}:{self.id2label[x[0]]}" for x in self.founds])
        right_counter = Counter(
            [f"{x[0]}:{self.id2label[x[0]]}" for x in self.rights])
        total_origin = 0
        total_found = 0
        total_right = 0
        for type_, count in origin_counter.items():
            category = type_.split(':')[1]
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {
                "acc": round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'right': right,
                'found': found,
                'origin': origin
            }
            if self.ignore_categories and category in self.ignore_categories:
                pass
            else:
                total_origin += origin
                total_found += found
                total_right += right
        if self.ignore_categories:
            origin = total_origin
            found = total_found
            right = total_right
        else:
            origin = len(self.origins)
            found = len(self.founds)
            right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)

        return {
            'acc': precision,
            'recall': recall,
            'f1': f1,
            'right': right,
            'found': found,
            'origin': origin
        }, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([
            pre_entity for pre_entity in pred_subject
            if pre_entity in true_subject
        ])


MODEL_CLASSES = {
    #  'bert': (BertConfig, BertPnForNer, CNerTokenizer),
    'bert': (BertConfig, BertPnForNer, BertTokenizer),
    #  'bert': (BertConfig, BertSpanForNer, BertTokenizerFast),
}


def load_pretrained_tokenizer(args):
    #  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    #  tokenizer = tokenizer_class.from_pretrained(
    #      args.model_path,
    #      do_lower_case=args.do_lower_case,
    #      is_english=args.is_english,
    #      cache_dir=args.cache_dir if args.cache_dir else None,
    #  )
    from ..token_utils import HFTokenizer
    tokenizer = HFTokenizer(os.path.join(args.model_path, 'vocab.txt'),
                            lowercase=args.do_lower_case,
                            cc=args.cc)

    return tokenizer


def load_pretrained_model(args):
    # make sure only the first process in distributed training
    # will download model & vocab
    barrier_member_processes(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.model_path,
        num_labels=args.num_labels,
        loss_type=args.loss_type,
        cache_dir=args.cache_dir if args.cache_dir else None,
        soft_label=args.soft_label,
    )
    setattr(config, 'label2id', args.label2id)
    setattr(config, 'id2label', args.id2label)
    setattr(config, 'loss_type', args.loss_type)
    setattr(config, 'focalloss_gamma', args.focalloss_gamma)
    setattr(config, 'focalloss_alpha', args.focalloss_alpha)
    setattr(config, 'diceloss_weight', args.diceloss_weight)
    logger.info(f"model_path: {args.model_path}")
    logger.info(f"config:{config}")
    model = model_class.from_pretrained(
        args.model_path,
        from_tf=bool(".ckpt" in args.model_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # make sure only the first process in distributed training
    # will download model & vocab
    barrier_leader_process(args)

    return model


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    #  from .dataset import batch_to_input_data
    #  all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens, all_token_offsets, all_subjects = batch_to_input_data(
    #      batch)
    all_input_ids = torch.stack([e.input_ids for e in batch])
    all_attention_mask = torch.stack([e.attention_mask for e in batch])
    all_token_type_ids = torch.stack([e.token_type_ids for e in batch])
    all_labels = torch.stack([e.labels for e in batch])
    all_input_lens = torch.stack([e.input_len for e in batch])
    all_token_offsets = torch.stack([e.token_offsets for e in batch])
    all_subjects = [e.subjects for e in batch]

    max_len = max(all_input_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    all_token_offsets = all_token_offsets[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_input_lens, all_token_offsets, all_subjects


def extract_entity_from_logits(args,
                               preds,
                               lens,
                               confidence=0.5,
                               enable_nested_entities=False):
    num_labels = int(preds.shape[-1] / 2)
    num_tokens = lens[0]

    #  logger.info(f"preds: (shape: {preds.shape}) | {preds}")
    preds = preds.view(1, -1, num_labels * 2)
    preds = preds.detach().cpu().numpy()

    starts = []
    ends = []
    for i in range(num_labels):
        start = np.where(preds[0, :, 2 * i] > confidence)[0]
        end = np.where(preds[0, :, 2 * i + 1] > confidence)[0]

        start = np.array([x for x in start if x >= 0 and x < num_tokens])
        end = np.array([x for x in end if x >= 0 and x < num_tokens])

        starts.append(start)
        ends.append(end)
    #  logger.info(f"starts: {starts}")
    #  logger.info(f"ends: {ends}")
    #  logger.info(f"enable_nested_entities: {enable_nested_entities}")
    entities = []
    for n in range(num_labels):
        start = starts[n]
        end = ends[n]

        last_end_pos = -1
        for idx, i in enumerate(start):
            if i <= last_end_pos:
                continue
            jj = end[end >= i]
            for j in jj:
                if idx < len(start) - 1 and j >= start[idx + 1]:
                    break
                s = i  #- 1
                e = j  #- 1
                last_end_pos = e
                entities.append((n + 1, s, e))
                if not enable_nested_entities:
                    break

    entities = sorted(entities, key=lambda x: x[1])
    return entities


def load_model(args):
    model = load_pretrained_model(args)
    model.to(args.device)
    return model


def build_default_model(args):
    """
    自定义模型
    规格要求返回模型(model)、优化器(optimizer)、调度器(scheduler)三元组。
    """

    # -------- model --------
    model = load_pretrained_model(args)
    model.to(args.device)

    # -------- optimizer --------
    from transformers.optimization import AdamW
    optimizer_parameters = get_default_optimizer_parameters(
        model, args.weight_decay)
    optimizer = AdamW(optimizer_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon,
                      correct_bias=False)

    # -------- scheduler --------
    from transformers.optimization import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.total_steps * args.warmup_rate,
        num_training_steps=args.total_steps)

    return model, optimizer, scheduler


def init_labels(args, labels):

    args.ner_labels = labels
    args.id2label = {i + 1: label for i, label in enumerate(args.ner_labels)}
    args.label2id = {label: i + 1 for i, label in enumerate(args.ner_labels)}
    args.num_labels = len(args.label2id)

    logger.info(f"args.label2id: {args.label2id}")
    logger.info(f"args.id2label: {args.id2label}")
    logger.info(f"args.num_labels: {args.num_labels}")


class NerTrainer(Trainer):
    def __init__(self, args, ner_labels, build_model=None, tokenizer=None):
        super(NerTrainer, self).__init__(args)
        init_labels(args, ner_labels)
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = load_pretrained_tokenizer(args)

        if build_model is None:
            self.build_model = build_default_model
        else:
            self.build_model = build_model

        self.label2id = args.label2id
        self.collate_fn = collate_fn

    #  def examples_to_dataset(self, examples, max_seq_length):
    #      from .dataset import examples_to_dataset
    #      return examples_to_dataset(examples, self.label2id, self.tokenizer,
    #                                 max_seq_length)
    def encode_examples(self, examples, max_seq_length):
        from .dataset import encode_examples
        return encode_examples(examples, self.label2id, self.tokenizer,
                               max_seq_length)

    def batch_to_inputs(self, args, batch, known_labels=True):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3],
        }
        if args.model_type != "distilbert":
            # XLM and RoBERTa don"t use token_type_ids
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "xlnet"] else None)

        return inputs

    #  def generate_dataloader(self, args, dataset, batch_size, keep_order=True):
    #
    #      Sampler = SequentialSampler if keep_order else RandomSampler
    #      sampler = DistributedSampler(dataset) if is_multi_processes(
    #          args) else Sampler(dataset)
    #      dataloader = DataLoader(dataset,
    #                              sampler=sampler,
    #                              batch_size=batch_size,
    #                              collate_fn=collate_fn)
    #      return dataloader

    def on_eval_start(self, args, eval_features):
        self.metric = SpanEntityScore(args.id2label,
                                      ignore_categories=args.ignore_categories)
        pass

    #  def on_eval_step(self, args, eval_dataset, step, model, inputs, outputs):
    #  def on_eval_step(self, args, model, step, batch, batch_features):
    def on_eval_step(self, args, model, step, batch):
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens, all_token_offsets, all_subjects = batch

        eval_loss = 0.0
        num_eval_steps = 0
        for i in range(all_input_ids.size()[0]):
            inputs = {
                "input_ids": all_input_ids[i].view(1, -1),
                "attention_mask": all_attention_mask[i].view(1, -1),
            }
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use token_type_ids
                inputs["token_type_ids"] = (all_token_type_ids[i].view(
                    1, -1) if args.model_type in ["bert", "xlnet"] else None)
            token_offsets = all_token_offsets[i]

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss
            num_eval_steps += 1

            #  loss_sig = nn.Sigmoid()
            #  active_logits = logits.view(-1, args.num_labels * 2)
            #  activae_logits = loss_sig(active_logits)
            #  active_logits = active_logits**2
            #  preds = active_logits

            #  active_logits = get_active_logits(logits, args.num_labels)
            #  preds = active_logits
            preds = logits
            num_tokens = int(all_lens[i])

            T = all_subjects[i]
            R = extract_entity_from_logits(
                args,
                preds, [num_tokens],
                confidence=args.confidence,
                enable_nested_entities=args.enable_nested_entities)
            #  logger.info(f"R: {R}")
            #  logger.info(f"T: {T}")

            self.metric.update(true_subject=T, pred_subject=R)

        eval_loss = eval_loss / num_eval_steps
        eval_info, entity_info = self.metric.result()
        results = {f'{key}': value for key, value in eval_info.items()}
        results['loss'] = eval_loss

        return (eval_loss, ), results

    def on_predict_start(self, args, test_features):
        self.pred_results = []

    def on_predict_step(self, args, model, step, batch):
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens, all_token_offsets, all_subjects = batch

        for i in range(all_input_ids.size()[0]):
            inputs = {
                "input_ids": all_input_ids[i].view(1, -1),
                "attention_mask": all_attention_mask[i].view(1, -1),
            }
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use token_type_ids
                inputs["token_type_ids"] = (all_token_type_ids[i].view(
                    1, -1) if args.model_type in ["bert", "xlnet"] else None)
            token_offsets = all_token_offsets[i]

            outputs = model(**inputs)
            logits = outputs[1]

            #  loss_sig = nn.Sigmoid()
            #  active_logits = logits.view(-1, args.num_labels * 2)
            #  activae_logits = loss_sig(active_logits)
            #  active_logits = active_logits**2
            #  preds = active_logits

            #  active_logits = get_active_logits(logits, args.num_labels)
            #  preds = active_logits
            preds = logits
            num_tokens = int(all_lens[i])

            R = extract_entity_from_logits(
                args,
                preds, [num_tokens],
                confidence=args.confidence,
                enable_nested_entities=args.enable_nested_entities)

            if R:
                #  label_entities = [[args.id2label[x[0]], x[1], x[2]] for x in R]
                label_entities = [[
                    args.id2label[x[0]], token_offsets[x[1]][0].item(),
                    token_offsets[x[2]][-1].item() - 1
                ] for x in R]
                if args.ignore_categories:
                    label_entities = [
                        x for x in label_entities
                        if x[0] not in args.ignore_categories
                    ]
                label_entities = [
                    x for x in label_entities
                    if x[1] <= x[2] and x[1] >= 0 and x[2] >= 0
                ]
            else:
                label_entities = []

            #  logger.info(f"label_entities: {label_entities}")

            #  if i < 20:
            #      logger.info(f"{i}, label_entities: {label_entities}")

            #  logger.debug(f"{label_entities}")
            json_d = {}
            json_d['id'] = step
            #  tag_seq = [args.id2label[x] for x in preds]
            #  json_d['tag_seq'] = " ".join(tag_seq)
            json_d['entities'] = label_entities

            #  logger.debug(f"{json_d}")

            self.pred_results.append(json_d)

    def on_predict_end(self, args, test_features):
        return self.pred_results

    def on_eval_end(self, args, eval_features):
        from ...utils.ner_utils import get_ner_results
        results = get_ner_results(self.metric)
        return results
