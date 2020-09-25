#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from loguru import logger
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import BertConfig, BertTokenizerFast
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from sklearn.metrics import classification_report, roc_auc_score
import mlflow

from ..trainer import Trainer, get_default_optimizer_parameters
from ...losses import FocalLoss, DiceLoss
from ...utils import softmax, sigmoid, acc_and_f1
from ...utils.multiprocesses import barrier_leader_process, barrier_member_processes


def logits_to_preds(logits):
    preds = np.argmax(logits, axis=1)
    probs = softmax(logits)
    # 调整类别0的判断阈值
    #  preds = np.array([
    #      0 if x == 1 and prob[1] < 0.60 else x for prob, x in zip(probs, preds)
    #  ])
    #  preds = np.array([
    #      1 if x == 0 and prob[1] < 0.90 else x for prob, x in zip(probs, preds)
    #  ])

    return preds


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.focalloss_gamma = config.focalloss_gamma
        self.focalloss_alpha = config.focalloss_alpha
        self.diceloss_weight = config.diceloss_weight

        self.init_weights()

    #  @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits, ) + outputs[
            2:]  # add hidden states and attention if they are here

        #  if labels is not None:
        #  if labels is not None and any(labels):
        if labels is not None and torch.sum(labels) > 0:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:

                #  #  logger.debug(f"logits: {type(logits)}, {logits}")
                #  preds = logits_to_preds(logits.detach().cpu().numpy())
                #  preds = torch.tensor(preds).float().cuda()
                #  loss = FocalLoss(gamma=1.5, alpha=None)(preds.view(-1),
                #                                          labels.view(-1))
                #

                #  loss_fct = CrossEntropyLoss()
                #  loss = loss_fct(logits.view(-1, self.num_labels),
                #                  labels.view(-1))

                #  logger.debug(f"preds: {type(preds)}, {preds}")

                assert self.loss_type in [
                    'FocalLoss', 'DiceLoss', 'CrossEntropyLoss',
                    'BCEWithLogitsLoss'
                ]
                if self.loss_type == 'FocalLoss':
                    loss = FocalLoss(gamma=self.focalloss_gamma,
                                     alpha=self.focalloss_alpha)(logits.view(
                                         -1, self.num_labels), labels.view(-1))
                elif self.loss_type == 'DiceLoss':
                    loss = DiceLoss(weight=self.diceloss_weight)(logits.view(
                        -1, self.num_labels), labels.view(-1))
                elif self.loss_type == 'BCEWithLogitsLoss':
                    loss_fct = BCEWithLogitsLoss()
                    #  logger.debug(f"logits: {logits.shape}, {logits}")
                    #  logger.debug(f"labels: {labels.shape}, {labels}")
                    #  loss = loss_fct(logits, labels.float())
                    loss = loss_fct(logits.view(-1, self.num_labels),
                                    labels.view(-1, self.num_labels).float())
                else:
                    loss_fct = CrossEntropyLoss()
                    #  logger.warning(f"logits: {logits.shape}")
                    #  logger.warning(f"labels: {labels.shape}")
                    loss = loss_fct(logits.view(-1, self.num_labels),
                                    labels.view(-1))
                    #  loss = loss_fct(logits.view(-1, self.num_labels),
                    #                  labels.view(-1, self.num_labels))

                #  from ...losses import FocalLoss
                #  #  loss = FocalLoss(gamma=1.5,
                #  #                   alpha=None)(logits.view(-1, self.num_labels),
                #  #                               labels.view(-1))
                #  loss = FocalLoss(
                #      gamma=1.5,
                #      alpha=[
                #          #          # --- 0.9354 (10/10)
                #          #          #  0.5,
                #          #          #  0.40,
                #          #          #  0.10,
                #          #          #  0.05,
                #          #          #  0.5,
                #          #          #  0.30,
                #          #          #  0.25,
                #          #          #  0.35
                #          #          # --- 0.9383 (10/10)
                #          #          0.5,
                #          #          0.40,
                #          #          0.10,
                #          #          0.05,
                #          #          0.5,
                #          #          0.40,
                #          #          0.40,
                #          #          0.40
                #          #  # focalloss_sympotom --- 0.9423 (10/10)
                #          #  # online f1: 0.92590
                #          #  0.50,  #   10
                #          #  0.40,  #   52
                #          #  0.10,  # 1267
                #          #  0.05,  # 2550
                #          #  0.50,  #    7
                #          #  0.40,  #  147
                #          #  0.35,  #  867
                #          #  0.40,  #  100
                #          #  # focalloss_sympotom --- 0.9396 (10/10)
                #          #  0.50,  #   10
                #          #  0.40,  #   52
                #          #  0.10,  # 1267
                #          #  0.05,  # 2550
                #          #  0.50,  #    7
                #          #  0.40,  #  147
                #          #  0.38,  #  867
                #          #  0.40,  #  100
                #          #  # focalloss_disease --- 0.9407 (7/10)
                #          #  # online f1: 0.92330
                #          #  0.50,  #   10
                #          #  0.40,  #   52
                #          #  0.08,  # 1267
                #          #  0.05,  # 2550
                #          #  0.50,  #    7
                #          #  0.40,  #  147
                #          #  0.35,  #  867
                #          #  0.40,  #  100
                #          #  # focalloss_others --- 0.9396 (10/10)
                #          #  0.50,  #   10
                #          #  0.40,  #   52
                #          #  0.10,  # 1267
                #          #  0.05,  # 2550
                #          #  0.50,  #    7
                #          #  0.40,  #  147
                #          #  0.35,  #  867
                #          #  0.30,  #  100
                #          # focalloss_checks --- 0.9397 (8/10)
                #          #  # online f1: 0.92640
                #          0.50,  #   10
                #          0.40,  #   52
                #          0.10,  # 1267
                #          0.05,  # 2550
                #          0.50,  #    7
                #          0.38,  #  147
                #          0.35,  #  867
                #          0.40,  #  100
                #      ])(logits.view(-1, self.num_labels), labels.view(-1))


#

#  glue_labels = ['病毒', '细菌', '疾病', '药物', '医学专科', '检查科目', '症状', 'NoneType']
# 10,     52,    1267,  2550,  7,     147,   867,   100
# [0.200, 0.200, 0.025, 0.025, 0.200, 0.100, 0.050, 0.200]

#  from ...losses import CircleLoss
#  loss = CircleLoss(m=0.25,
#                    gamma=128)
#  loss = CircleLoss(scale=32, margin=0.25)(logits.view(
#      -1, self.num_labels), labels.view(-1))
#  outputs = (loss, ) + outputs
            return (loss, ) + outputs
        else:
            # (loss), logits, (hidden_states), (attentions)
            #  return (torch.tensor(0.0).cuda(), ) + outputs
            return (None, ) + outputs
            #  return outputs

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizerFast),
    #  'albert':
    #  (AlbertConfig, AlBertForSequenceClassification, BertTokenizerFast),
}


def load_pretrained_tokenizer(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        args.model_path,
        do_lower_case=args.do_lower_case,
        is_english=args.is_english,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    return tokenizer


def load_pretrained_model(args):
    # make sure only the first process in distributed training
    # will download model & vocab
    barrier_member_processes(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.model_path,
        num_labels=args.num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
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
        cache_dir=args.cache_dir if args.cache_dir else None)

    # make sure only the first process in distributed training
    # will download model & vocab
    barrier_leader_process(args)

    return model


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
                      correct_bias=False)

    # -------- scheduler --------
    from transformers.optimization import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.total_steps * args.warmup_rate,
        num_training_steps=args.total_steps)

    return model, optimizer, scheduler


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(
        torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    #  all_labels = all_labels[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


class ClassReporter(object):
    def __init__(self, target_names=None, labels=None):
        logger.debug(f"target_names: {target_names}")
        logger.debug(f"labels: {labels}")
        self.target_names = target_names
        self.labels = labels

    def __call__(self, output, target):
        #  _, y_pred = torch.max(output.data, 1)
        #  y_pred = y_pred.cpu().numpy()
        #  y_true = target.cpu().numpy()

        #  y_pred = output.cpu().numpy()
        #  y_true = target.cpu().numpy()

        y_pred = output
        y_true = target

        #  print(y_pred, y_true, self.labels, self.target_names)
        #  if len(y_true.shape) == 1:
        classify_report = classification_report(y_true,
                                                y_pred,
                                                labels=self.labels,
                                                target_names=self.target_names)
        logger.info(f"classify_report:\n{classify_report}")
        #  else:
        #      for i, label in enumerate(self.labels):
        #          auc = roc_auc_score(y_score=y_pred[:, i], y_true=y_true[:, i])
        #          print(f"{label} - auc: {auc:.4f}")

        return classify_report


def init_labels(args, labels):
    args.id2label = {i: label for i, label in enumerate(labels)}
    args.label2id = {label: i for i, label in enumerate(labels)}
    args.num_labels = len(args.label2id)

    logger.info(f"args.label2id: {args.label2id}")
    logger.info(f"args.id2label: {args.id2label}")
    logger.info(f"args.num_labels: {args.num_labels}")


class GlueTrainer(Trainer):
    def __init__(self, args, glue_labels, build_model=None, tokenizer=None):
        super(GlueTrainer, self).__init__(args)
        init_labels(args, glue_labels)

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

        self.logits = None
        self.preds = None
        self.probs = None

        self.class_reporter = ClassReporter(
            target_names=[x for x in self.label2id.keys()],
            labels=[x for x in self.label2id.values()])

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

    def examples_to_dataset(self, examples, max_seq_length):
        from .dataset import examples_to_dataset
        return examples_to_dataset(examples, self.label2id, self.tokenizer,
                                   max_seq_length)

    def batch_to_inputs(self, args, batch, known_labels=True):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3]
        }
        if args.model_type != "distilbert":
            # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            inputs["token_type_ids"] = (batch[2] if args.model_type in [
                "bert", "xlnet", "albert"
            ] else None)
        return inputs

    def on_eval_start(self, args, eval_dataset):
        self.logits = None
        self.preds = None
        self.out_label_ids = None
        self.results = {}
        self.probs = None

    #  def on_eval_step(self, args, eval_dataset, step, model, inputs, outputs):
    def on_eval_step(self, args, model, step, batch, batch_features):
        input_ids, attention_mask, _, label_ids = batch
        outputs = self.on_train_step(args, model, step, batch)
        eval_loss, logits = outputs[:2]
        if self.logits is None:
            self.logits = logits.detach().cpu().numpy()
            self.out_label_ids = label_ids.detach().cpu().numpy()
        else:
            self.logits = np.append(self.logits,
                                    logits.detach().cpu().numpy(),
                                    axis=0)
            self.out_label_ids = np.append(self.out_label_ids,
                                           label_ids.detach().cpu().numpy(),
                                           axis=0)

        self.preds = np.argmax(self.logits, axis=1)
        self.probs = softmax(self.logits)

        # 调整类别0的判断阈值
        #  self.preds = np.array([
        #      0 if x == 1 and prob[1] < 0.60 else x
        #      for prob, x in zip(self.probs, self.preds)
        #  ])
        self.preds = logits_to_preds(self.logits)
        #  logger.debug(f"self.preds: {self.preds.shape}, {self.preds}")
        #  logger.debug(
        #      f"self.out_label_ids: {self.out_label_ids.shape}, {self.out_label_ids}, {np.argmax(self.out_label_ids, axis=1)}"
        #  )

        if len(self.out_label_ids.shape) == 1:
            result = acc_and_f1(self.preds, self.out_label_ids)
        else:
            result = acc_and_f1(self.preds,
                                np.argmax(self.out_label_ids, axis=1))

        if args.do_experiment:
            #  mlflow.log_metric('loss', eval_loss.item())
            for key, value in result.items():
                mlflow.log_metric(key, value)

        self.results.update(result)

        return (eval_loss, ), self.results

    def on_eval_end(self, args, eval_dataset):
        if len(self.out_label_ids.shape) == 1:
            self.preds = np.argmax(self.logits, axis=1)
        else:
            self.preds = np.array([x > 0.5
                                   for x in sigmoid(self.logits)]).astype(int)
        # for regessions
        #  self.preds = np.squeeze(self.preds)

        #  logger.debug(f"self.preds: {self.preds.shape}, {self.preds}")
        #  logger.debug(
        #      f"self.out_label_ids: {self.out_label_ids.shape}, {self.out_label_ids}"
        #  )
        self.class_reporter(self.preds, self.out_label_ids)

        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Batch size = {args.eval_batch_size}")
        logger.info(f"******** Eval results ********")
        for key in self.results.keys():
            logger.info(f" dev: {key} = {self.results[key]:.6f}")

        return self.results

    def on_predict_start(self, args, test_dataset):
        self.logits = None

        self.pred_results = None
        self.probs = None

    #  def on_predict_step(self, args, test_dataset, step, model, inputs,
    #                      outputs):
    def on_predict_step(self, args, model, step, batch):
        inputs = self.batch_to_inputs(args, batch, known_labels=False)
        outputs = model(**inputs)

        #  logger.debug(f"inputs: {inputs}")
        #  logger.debug(f"outputs: {outputs}")
        logits = outputs[1]
        #  logits = outputs[0]
        if self.logits is None:
            self.logits = logits.detach().cpu().numpy()
        else:
            self.logits = np.append(self.logits,
                                    logits.detach().cpu().numpy(),
                                    axis=0)

    def on_predict_end(self, args, test_dataset):
        self.pred_results = np.argmax(self.logits, axis=1)
        self.probs = softmax(self.logits)

        #  self.pred_results = np.array([
        #      0 if x == 1 and prob[1] < 0.60 else x
        #      for prob, x in zip(self.probs, self.pred_results)
        #  ])
        self.pred_results = logits_to_preds(self.logits)

        logger.debug(f"pred_results: {self.pred_results}")
        logger.debug(f"probs: {self.probs}")
