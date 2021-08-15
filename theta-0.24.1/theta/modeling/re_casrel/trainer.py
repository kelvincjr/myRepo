#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (AutoConfig, AutoModelForTokenClassification,
                          BertConfig, BertTokenizer, BertTokenizerFast)
from transformers.modeling_bert import BertModel, BertPreTrainedModel
#from transformers.models.bert import BertModel, BertPreTrainedModel

from ...losses import DiceLoss, FocalLoss, LabelSmoothingCrossEntropy
from ...utils.multiprocesses import (barrier_leader_process,
                                     barrier_member_processes,
                                     is_multi_processes)
from ..models.linears import PoolerEndLogits, PoolerStartLogits
from ..trainer import Trainer, get_default_optimizer_parameters
from .dataset import batch_to_input_data

BertLayerNorm = torch.nn.LayerNorm


def get_active_logits(logits, num_labels):
    # Only keep active parts of the loss
    loss_sig = nn.Sigmoid()
    active_logits = logits.view(-1, num_labels * 2)
    active_logits = loss_sig(active_logits)
    active_logits = active_logits**2

    return active_logits


def batch_gather(data: torch.Tensor, index: torch.Tensor):
    index = index.unsqueeze(-1)
    index = index.expand(data.size()[0], index.size()[1], data.size()[2])
    #  logger.info(f"batch_gather() index.size(): {index.size()}")
    return torch.gather(data, 1, index)


def extract_subject_1(output, seed_subs):
    """根据seed_subs从output中取出subject的向量表征
    """
    output = output.detach().cpu()
    seed_subs = seed_subs.detach().cpu()

    #  logger.warning(f"seed_subs: {seed_subs}")
    #  logger.warning(
    #      f"output.shape: {output.shape}, seed_subs.shape: {seed_subs.shape}")

    start = batch_gather(output, seed_subs[:, :1])
    end = batch_gather(output, seed_subs[:, 1:])
    # so_res = merge_function([start,end])
    # subject = torch.cat([start, end], 2)
    #  output.cuda()
    #  seed_subs.cuda()
    #  start.cuda()
    #  end.cuda()
    return start, end


    # return so_res
class BertCasrelForRE(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCasrelForRE, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.focalloss_gamma = config.focalloss_gamma
        self.focalloss_alpha = config.focalloss_alpha
        self.diceloss_weight = config.diceloss_weight

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.classifier = nn.Linear(config.hidden_size, 2)

        self.obj_classifier = nn.Linear(config.hidden_size,
                                        self.num_labels * 2)
        self.sub_pos_emb = nn.Embedding(256, config.hidden_size)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        sub_labels=None,
        seed_subs=None,
        obj_labels=None,
        do_forward_sub=True,
        do_forward_obj=True,
    ):
        batch_size = input_ids.size(0)
        outputs_bert = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        #  logger.info(f"outputs_bert: {outputs_bert}")
        #  for i, x in enumerate(outputs_bert):
        #      logger.info(f"{i}: {x.shape}")
        #  logger.info(
        #      f"do_forward_sub: {do_forward_sub}, do_forward_obj: {do_forward_obj}"
        #  )
        sequence_output = outputs_bert[0]
        sequence_output = self.dropout(sequence_output)

        outputs = self.forward_sub(sequence_output, attention_mask, sub_labels,
                                   batch_size)
        if do_forward_obj:
            if sub_labels is not None and obj_labels is not None:
                loss = outputs[0]
            else:
                loss = None
            outputs_obj = self.forward_obj(outputs_bert, sequence_output,
                                           attention_mask, seed_subs,
                                           obj_labels, batch_size, loss)
            return outputs_obj  # (loss), scores, (hidden_states), (attentions)
        else:
            return outputs

    def forward_sub(self, sequence_output, attention_mask, sub_labels,
                    batch_size):

        logits = self.classifier(sequence_output)
        outputs = (logits,
                   )  # add hidden states and attention if they are here
        #  logger.warning(f"logits.shape: {logits.shape}")
        loss_fct = BCELoss(reduction='none')
        # loss_fct = BCEWithLogitsLoss(reduction='none')
        loss_sig = nn.Sigmoid()
        # Only keep active parts of the loss
        #  active_logits = logits.view(-1, self.num_labels)
        active_logits = logits.view(-1, 2)
        active_logits = loss_sig(active_logits)
        active_logits = active_logits**2
        #  logger.warning(f"active_logits.shape: {active_logits.shape}")
        if sub_labels is not None:
            #  active_labels = labels.view(-1, self.num_labels).float()
            active_labels = sub_labels.view(-1, 2).float()
            #  logger.warning(
            #      f"active_logits.shape: {active_logits.shape}, active_labels.shape: {active_labels.shape}"
            #  )
            loss = loss_fct(active_logits, active_labels)
            # loss = loss.view(-1,sequence_output.size()[1],2)
            loss = loss.view(batch_size, -1, 2)
            loss = torch.mean(loss, 2)
            loss = torch.sum(attention_mask * loss) / torch.sum(attention_mask)
            outputs = (loss, ) + outputs
        else:
            outputs = (torch.tensor(0.0), active_logits)
            #  outputs = (torch.tensor(0.0), logits)

        return outputs
        #  logits = self.classifier(sequence_output)
        #  outputs = (logits,
        #             )  # add hidden states and attention if they are here
        #  loss_fct = BCELoss(reduction='none')
        #  # loss_fct = BCEWithLogitsLoss(reduction='none')
        #  loss_sig = nn.Sigmoid()
        #  # Only keep active parts of the loss
        #  active_logits = logits.view(-1, self.num_labels)
        #  active_logits = loss_sig(active_logits)
        #  active_logits = active_logits**2
        #  if sub_labels is not None:
        #      active_labels = sub_labels.view(-1, self.num_labels).float()
        #      loss = loss_fct(active_logits, active_labels)
        #      # loss = loss.view(-1,sequence_output.size()[1],2)
        #      loss = loss.view(batch_size, -1, 2)
        #      loss = torch.mean(loss, 2)
        #      loss = torch.sum(attention_mask * loss) / torch.sum(attention_mask)
        #      outputs = (loss, ) + outputs
        #  else:
        #      outputs = active_logits
        #      outputs = (torch.tensor(0.0).cuda(), ) + active_logits
        #
        #  return outputs

    def forward_obj(self,
                    outputs_bert,
                    sequence_output,
                    attention_mask,
                    seed_subs,
                    obj_labels,
                    batch_size,
                    loss=None):

        #  logger.info(f"1-obj_labels: {obj_labels}")
        hidden_states = outputs_bert[2][-2]
        hidden_states_1 = outputs_bert[2][-3]
        # hidden_states = self.dropout(hidden_states)
        loss_obj = BCELoss(reduction='none')
        loss_sig = nn.Sigmoid()
        #sub_pos_start = self.sub_pos_emb(seed_subs[:, :1]).cuda()
        #sub_pos_end = self.sub_pos_emb(seed_subs[:, 1:]).cuda()
        sub_pos_start = self.sub_pos_emb(seed_subs[:, :1]).cpu()
        sub_pos_end = self.sub_pos_emb(seed_subs[:, 1:]).cpu()

        #  logger.info(f"1.0-seed_subs: {seed_subs}")
        #  logger.info(f"1.0-obj_labels: {obj_labels}")

        subject_start_last, subject_end_last = extract_subject_1(
            sequence_output, seed_subs)

        #  logger.info(f"1.0.1-seed_subs: {seed_subs}")
        #  logger.info(f"1.0.1-obj_labels: {obj_labels}")

        subject_start_1, subject_end_1 = extract_subject_1(
            hidden_states_1, seed_subs)

        #  logger.info(f"1.0.2-obj_labels: {obj_labels}")

        subject_start, subject_end = extract_subject_1(hidden_states,
                                                       seed_subs)
        #  logger.info(f"1.1-obj_labels: {obj_labels}")
        # subject = extract_subject(hidden_states, seed_subs)
        # subject_start = subject_start.cuda()
        # subject_end = subject_end.cuda())

        #subject_start_last = subject_start_last.cuda()
        #subject_end_last = subject_end_last.cuda()
        #subject_start_1 = subject_start_1.cuda()
        #subject_end_1 = subject_end_1.cuda()
        #subject_start = subject_start.cuda()
        #subject_end = subject_end.cuda()
        subject_start_last = subject_start_last.cpu()
        subject_end_last = subject_end_last.cpu()
        subject_start_1 = subject_start_1.cpu()
        subject_end_1 = subject_end_1.cpu()
        subject_start = subject_start.cpu()
        subject_end = subject_end.cpu()

        #  logger.info(
        #      f"sub_pos_start.shape: {sub_pos_start.shape}, sub_pos_end.shape: {sub_pos_end.shape}, subject_start.shape: {subject_start.shape}, subject_end.shape: {subject_end.shape}"
        #  )
        #  logger.info(f"subject_start_last.shape: {subject_start_last.shape}")
        #  logger.info(
        #      f"subject_start_1.shape: {subject_start_1.shape}, subject_end_1.shape: {subject_end_1.shape}"
        #  )
        #subject = (sub_pos_start + subject_start + sub_pos_end + subject_end +
                   #subject_start_last + subject_start_1 + subject_end_1 +
                   #subject_end_1).cuda()
        subject = (sub_pos_start + subject_start + sub_pos_end + subject_end +
                   subject_start_last + subject_start_1 + subject_end_1 +
                   subject_end_1).cpu()
        # subject = extract_subject(sequence_output, seed_subs).cuda()
        batch_token_ids_obj = torch.add(hidden_states, subject)
        batch_token_ids_obj = self.LayerNorm(batch_token_ids_obj)
        batch_token_ids_obj = self.dropout(batch_token_ids_obj)

        #  logger.info(f"1.2-obj_labels: {obj_labels}")

        # batch_token_ids_obj = F.dropout(batch_token_ids_obj,p=0.5)
        batch_token_ids_obj = self.relu(self.linear(batch_token_ids_obj))
        batch_token_ids_obj = self.dropout(batch_token_ids_obj)
        # batch_token_ids_obj = F.dropout(batch_token_ids_obj,p=0.4)
        obj_logits = self.obj_classifier(batch_token_ids_obj)
        #  logger.warning(f"obj_labels.shape: {obj_labels.shape}")
        #  logger.info(f"obj_logits.shape: {obj_logits.shape}")

        obj_logits = obj_logits.view(-1, 2)
        obj_logits = loss_sig(obj_logits)
        obj_logits = obj_logits**4
        obj_outputs = (obj_logits, )

        #  active_logits = logits.view(-1, 2)
        #  active_logits = loss_sig(active_logits)
        #  active_logits = active_logits**2
        #  logger.info(f"2-obj_labels: {obj_labels}")
        if obj_labels is not None:
            #  obj_logits = obj_logits.view(-1,2)
            #  obj_logits = loss_sig(obj_logits)
            #  obj_logits = obj_logits**4
            #  obj_outputs = (obj_logits, )

            #  logger.warning(f"obj_labels: {obj_labels}")
            #  logger.info(
            #      f"obj_labels: {obj_labels.shape}, obj_logits: {obj_logits.shape}"
            #  )
            # obj_loss = loss_obj(obj_logits.view(-1, hidden_states.size()[1], self.obj_labels // 2, 2), obj_labels.float())
            obj_loss = loss_obj(
                #  obj_logits.view(batch_size, -1, self.num_labels // 2, 2),
                obj_logits.view(batch_size, -1, self.num_labels, 2),
                obj_labels.float())

            #  logger.warning(f"obj_loss: {obj_loss.shape}")
            #  logger.warning(f"obj_loss: {obj_loss[0]}")
            #  logger.warning(f"obj_loss: {obj_loss.shape}, {obj_loss}")
            obj_loss = torch.sum(torch.mean(obj_loss, 3), 2)
            #  logger.info(f"obj_loss: {obj_loss.shape}, {obj_loss}")
            obj_loss = torch.sum(
                obj_loss * attention_mask) / torch.sum(attention_mask)
            #  logger.info(f"obj_loss: {obj_loss.shape}, {obj_loss}")
            s_o_loss = torch.add(obj_loss, loss)
            #  logger.info(f"s_o_loss: {s_o_loss.shape}, {s_o_loss}")
            outputs_obj = (s_o_loss, ) + obj_outputs
        else:
            # outputs_obj = obj_logits.view(-1,hidden_states.size()[1],self.obj_labels // 2 ,2)
            outputs_obj = obj_logits.view(batch_size, -1, self.num_labels, 2)
            #  outputs_obj = obj_logits.view(batch_size, -1, self.num_labels // 2,
            #                                2)

        return outputs_obj


def extract_spoes(args, model, text, confidence=0.5):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer_k.tokenize(text, max_length=maxlen)
    # tokens_1 = tokenizer.tokenize(text,max_length=maxlen)
    mapping = tokenizer_k.rematch(text, tokens)
    token_ids, segment_ids = tokenizer_k.encode(text, max_length=maxlen)
    # token_ids_1 = tokenizer.encode_plus(text,max_length=maxlen)['input_ids']

    #  token_ids_1 = token_ids
    #  segment_ids_1 = segment_ids
    #token_ids = torch.tensor([token_ids]).cuda()
    #segment_ids = torch.tensor([segment_ids]).cuda()
    token_ids = torch.tensor([token_ids]).cpu()
    segment_ids = torch.tensor([segment_ids]).cpu()

    subject_preds = model(input_ids=token_ids,
                          token_type_ids=segment_ids,
                          do_forward_obj=False)

    subjects = extract_subjects_from_logits(args,
                                            subject_preds,
                                            confidence=confidence)
    spoes = extract_predicate_objects_from_logits(args,
                                                  model,
                                                  subjects,
                                                  mapping,
                                                  confidence=confidence)

    return spoes


def extract_subjects_from_logits(
    args,
    subject_preds,
    confidence=0.5,
):
    # 抽取subject
    subject_preds = subject_preds.view(1, -1, 2)
    subject_preds = subject_preds.detach().cpu().numpy()
    #  logger.info(f"subject_preds.shape: {subject_preds.shape}")

    start = np.where(subject_preds[0, :, 0] > confidence)[0]
    end = np.where(subject_preds[0, :, 1] > confidence)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    return subjects


def extract_predicate_objects_from_logits(args,
                                          model,
                                          subjects,
                                          input_ids,
                                          attention_mask,
                                          token_type_ids,
                                          token2char,
                                          lens,
                                          confidence=0.5):

    #  logger.info(f"input_ids: {input_ids}")

    input_ids = input_ids.detach().cpu().numpy().tolist()
    attention_mask = attention_mask.detach().cpu().numpy().tolist()
    token_type_ids = token_type_ids.detach().cpu().numpy().tolist()

    sub_input_ids = np.repeat(input_ids, len(subjects), 0)
    #  sub_input_ids = np.repeat([input_ids], len(subjects), 0)
    sub_attention_mask = np.repeat([attention_mask], len(subjects), 0)
    sub_token_type_ids = np.repeat(token_type_ids, len(subjects), 0)
    #  sub_token_type_ids = np.repeat([token_type_ids], len(subjects), 0)

    #sub_input_ids = torch.tensor(sub_input_ids).cuda()
    #sub_attention_mask = torch.tensor(sub_attention_mask).cuda()
    #sub_token_type_ids = torch.tensor(sub_token_type_ids).cuda()

    #subjects = torch.tensor(subjects).cuda()
    sub_input_ids = torch.tensor(sub_input_ids).cpu()
    sub_attention_mask = torch.tensor(sub_attention_mask).cpu()
    sub_token_type_ids = torch.tensor(sub_token_type_ids).cpu()

    subjects = torch.tensor(subjects).cpu()
    # 传入subject，抽取object和predicate
    #  logger.info(
    #      f"sub_input_ids.shape: {sub_input_ids.shape}, sub_token_type_ids.shape: {sub_token_type_ids.shape}"
    #  )
    #  #  sub_attention_mask.shape: {sub_attention_mask.shape}
    #  logger.info(f"sub_input_ids: {sub_input_ids}")
    #  logger.info(f"sub_token_type_ids: {sub_token_type_ids}")
    #  logger.info(f"subjects: {subjects}")
    object_preds = model(input_ids=sub_input_ids,
                         attention_mask=sub_attention_mask,
                         token_type_ids=sub_token_type_ids,
                         seed_subs=subjects,
                         do_forward_sub=True,
                         do_forward_obj=True)
    #  logger.warning(f"object_preds:{object_preds.shape}, {object_preds}")

    text_len = lens[0].item()
    spoes = []
    object_preds = object_preds.detach().cpu().numpy()
    for subject, object_pred in zip(subjects, object_preds):
        start = np.where(object_pred[:, :, 0] > confidence)
        end = np.where(object_pred[:, :, 1] > confidence)
        #  logger.info(f"text_len: {text_len}, start: {start}, end: {end}")
        #  start = np.array([x for x in start if x >= 0 and x < text_len])
        #  end = np.array([x for x in end if x >= 0 and x < text_len])

        last_end = -1
        for _start, predicate1 in zip(*start):
            if _start <= last_end:
                continue
            if _start >= text_len:
                continue
            found = False
            for _end, predicate2 in zip(*end):
                if _end >= text_len:
                    continue
                if _start <= _end and predicate1 == predicate2:
                    s_subject, e_subject = subject
                    #  s_subject = token2char[s_subject][0].item()
                    #  e_subject = token2char[e_subject][-1].item()
                    #  _start = token2char[_start][0].item()
                    #  _end = token2char[_end][-1].item()
                    if s_subject >= 0 and e_subject >= 0 and _start >= 0 and _end >= 0:
                        #  logger.info(
                        #      f"spo: ({s_subject}:{e_subject}), {predicate1}, ({_start}:{_end})"
                        #  )
                        found = True
                        spoes.append(((s_subject, e_subject), predicate1,
                                      (_start, _end)))
                        last_end = _end

                    #  if len(mapping[subject[0]]) > 0 and len(
                    #          mapping[subject[1]]) > 0 and len(
                    #              mapping[_start]) > 0 and len(
                    #                  mapping[_end]) > 0:
                    #      spoes.append(((mapping[subject[0]][0],
                    #                     mapping[subject[1]][-1]), predicate1,
                    #                    (mapping[_start][0], mapping[_end][-1])))
                    break
            #  if found:
            #      break
    return spoes
    #  return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
    #          for s, p, o, in spoes]


def combine_spoes(spoes):
    """合并SPO成官方格式
    """
    new_spoes = {}
    for s, p, o in spoes:
        p1, predicate, p2 = p.split('_')
        if (s, p1) in new_spoes:
            new_spoes[(s, p1)][p2] = o
        else:
            new_spoes[(s, p1)] = {p2: o}

    return [(k[0], k[1], v) for k, v in new_spoes.items()]


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo, tokenizer_k):
        self.spox = (
            tuple(tokenizer_k.tokenize(spo[0])),
            spo[1],
            tuple(
                sorted([(k, tuple(tokenizer_k.tokenize(v)))
                        for k, v in spo[2].items()])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(data, tokenizer_k):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred_2.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:

        R = combine_spoes(extract_spoes(d['text']))
        T = combine_spoes(d['spo_list'])
        R = set([SPO(spo, tokenizer_k) for spo in R])
        T = set([SPO(spo, tokenizer_k) for spo in T])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' %
                             (f1, precision, recall))
        s = json.dumps(
            {
                'text': d['text'],
                'spo_list': list(T),
                'spo_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            },
            ensure_ascii=False,
            indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall


class SpanEntityScore(object):
    def __init__(self, id2label):
        self.id2label = id2label
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
        origin_counter = Counter([self.id2label[x[0]] for x in self.origins])
        found_counter = Counter([self.id2label[x[0]] for x in self.founds])
        right_counter = Counter([self.id2label[x[0]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {
                "acc": round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4)
            }
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([
            pre_entity for pre_entity in pred_subject
            if pre_entity in true_subject
        ])


MODEL_CLASSES = {
    #  'bert': (BertConfig, BertCasrelForRE, CNerTokenizer),
    'bert': (BertConfig, BertCasrelForRE, BertTokenizer),
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
                            lowercase=args.do_lower_case)
    #  from tokenizers import BertWordPieceTokenizer
    #  tokenizer = BertWordPieceTokenizer(
    #      os.path.join(args.model_path, "vocab.txt"))
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
        output_hidden_states=True,
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
    all_input_ids, all_input_mask, all_segment_ids, all_sub_labels, all_seed_subs, all_obj_labels, all_lens, all_token_offsets, all_texts, all_spo_lists = batch_to_input_data(
        batch)
    #  all_input_ids, all_input_mask, all_segment_ids, all_sub_labels, all_seed_subs, all_obj_labels, all_lens, all_token2char, all_token_offsets = map(
    #      torch.stack, zip(*batch))

    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    all_sub_labels = all_sub_labels[:, :max_len]
    all_obj_labels = all_obj_labels[:, :max_len]

    return all_input_ids, all_input_mask, all_segment_ids, all_sub_labels, all_seed_subs, all_obj_labels, all_lens, all_token_offsets, all_texts, all_spo_lists


def extract_entity_from_logits(args,
                               preds,
                               lens,
                               confidence=0.5,
                               enable_nested_entities=False):
    num_labels = int(preds.shape[-1] / 2)
    text_len = lens[0]

    #  logger.info(f"preds: (shape: {preds.shape})")  # |{preds})")
    preds = preds.view(1, -1, num_labels * 2)
    preds = preds.detach().cpu().numpy()

    starts = []
    ends = []
    for i in range(num_labels):
        start = np.where(preds[0, :, 2 * i] > confidence)[0]
        end = np.where(preds[0, :, 2 * i + 1] > confidence)[0]

        start = np.array([x for x in start if x >= 0 and x < text_len])
        end = np.array([x for x in end if x >= 0 and x < text_len])

        starts.append(start)
        ends.append(end)
    #  logger.info(f"starts: {starts}")
    #  logger.info(f"ends: {ends}")
    entities = []
    for n in range(num_labels):
        start = starts[n]
        end = ends[n]
        for idx, i in enumerate(start):
            jj = end[end >= i]
            for j in jj:
                if idx < len(start) - 1 and j >= start[idx + 1]:
                    break
                entities.append((n + 1, i - 1, j - 1))
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
    args.id2label = {i: label for i, label in enumerate(args.ner_labels)}
    args.label2id = {label: i for i, label in enumerate(args.ner_labels)}
    args.num_labels = len(args.label2id)

    logger.info(f"args.label2id: {args.label2id}")
    logger.info(f"args.id2label: {args.id2label}")
    logger.info(f"args.num_labels: {args.num_labels}")


def evaluate_spo_list(pred, target):
    """
    []((sub_start, sub_mention), predicate, (obj_start, obj_mention)), ...]
    """

    R = set(pred)
    T = set(target)
    X = len(R & T)
    Y = len(R)
    Z = len(T)

    return X, Y, Z


class SpoTrainer(Trainer):
    def __init__(self, args, ner_labels, build_model=None, tokenizer=None):
        super(SpoTrainer, self).__init__(args)
        init_labels(args, ner_labels)
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = load_pretrained_tokenizer(args)

        #  from bert4keras.tokenizers import Tokenizer
        #  self.tokenizer_k = Tokenizer(os.path.join(args.model_path,
        #                                            'vocab.txt'),
        #                               do_lower_case=True)

        if build_model is None:
            self.build_model = build_default_model
        else:
            self.build_model = build_model

        self.label2id = args.label2id
        self.collate_fn = collate_fn

    def examples_to_dataset(self, examples, max_seq_length):
        from .dataset import examples_to_dataset
        return examples_to_dataset(examples, self.label2id, self.tokenizer,
                                   max_seq_length)

    def encode_examples(self, examples, max_seq_length):
        from .dataset import encode_examples
        return encode_examples(examples, self.label2id, self.tokenizer,
                               max_seq_length)

    def batch_to_inputs(self, args, batch, known_labels=True):
        #  inputs = {
        #      "input_ids": torch.stack([e.input_ids for e in batch]),
        #      "attention_mask": torch.stack([e.input_mask for e in batch]),
        #      "sub_labels": torch.stack([e.sub_labels for e in batch]),
        #      "seed_subs": torch.stack([e.seed_subs for e in batch]),
        #      "obj_labels": torch.stack([e.obj_labels for e in batch]),
        #  }
        #  if args.model_type != "distilbert":
        #      # XLM and RoBERTa don"t use segment_ids
        #      inputs["token_type_ids"] = (torch.stack([
        #          e.segment_ids for e in batch
        #      ]) if args.model_type in ["bert", "xlnet"] else None)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "sub_labels": batch[3],
            "seed_subs": batch[4],
            "obj_labels": batch[5],
        }
        if args.model_type != "distilbert":
            # XLM and RoBERTa don"t use segment_ids
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

    def on_eval_start(self, args, eval_dataset):
        self.eval_results = []
        self.metric = SpanEntityScore(args.id2label)
        self.X, self.Y, self.Z = 1e-10, 1e-10, 1e-10

    #  def on_eval_step(self, args, model, step, batch, batch_features):
    def on_eval_step(self, args, model, step, batch):

        all_input_ids, all_input_mask, all_segment_ids, all_sub_labels, all_seed_subs, all_obj_labels, all_lens, all_token_offsets, all_texts, all_spo_lists = batch

        X = self.X
        Y = self.Y
        Z = self.Z
        eval_loss = 0.0
        num_eval_steps = 0
        for i in range(all_input_ids.size()[0]):
            inputs = {
                "input_ids": all_input_ids[i].view(1, -1),
                "attention_mask": all_input_mask[i].view(1, -1),
                #  "sub_labels": all_sub_labels[i].view(1, -1),
                #  "seed_subs": all_seed_subs[i].view(1, -1),
                #  "obj_labels": all_obj_labels[i].view(1, -1),
                "do_forward_obj": False,
            }
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (all_segment_ids[i].view(
                    1, -1) if args.model_type in ["bert", "xlnet"] else None)
            token_offsets = all_token_offsets[i]
            text = all_texts[i]
            #  logger.warning(f"text: {len(text)} {text}")
            spo_list = all_spo_lists[i]

            outputs = model(**inputs)
            #  logger.warning(f"ouptuts: {outputs}")

            batch_eval_loss, subject_preds = outputs[:2]
            eval_loss += batch_eval_loss
            num_eval_steps += 1

            #  logger.info(f"subject_preds.shape: {subject_preds.shape}")
            subjects = extract_subjects_from_logits(args,
                                                    subject_preds,
                                                    confidence=args.confidence)
            #  logger.warning(f"text: {text}")
            #  logger.info(f"spo_list: {spo_list}")
            pred_subjects = []
            for sub in subjects:
                s_text = token_offsets[sub[0] - 1][0]
                e_text = token_offsets[sub[1] - 1][-1]
                pred_subjects.append(text[s_text:e_text])
            #  logger.info(f"pred_subjects: {pred_subjects}")
            pred_spoes = []
            if subjects:
                spoes = extract_predicate_objects_from_logits(
                    args,
                    model,
                    subjects,
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    inputs['token_type_ids'],
                    token_offsets,
                    all_lens[i:i + 1],
                    confidence=args.confidence / 4)
                #  logger.info(f"Eval spoes: {spoes}")
                for spo in spoes:
                    (s_sub, e_sub), predicate, (s_obj, e_obj) = spo
                    s_sub_text = token_offsets[s_sub - 1][0].item()
                    e_sub_text = token_offsets[e_sub - 1][-1].item()
                    s_obj_text = token_offsets[s_obj - 1][0].item()
                    e_obj_text = token_offsets[e_obj - 1][-1].item()
                    if e_sub_text - s_sub_text >= 2 and e_obj_text - s_obj_text >= 2:
                        #  logger.info(f"{text[s_sub_text:e_sub_text]} , {args.id2label[predicate]}, {text[s_obj_text:e_obj_text]}")
                        pred_predicate = args.id2label[predicate]
                        s_mention = text[s_sub_text:e_sub_text]
                        pred_subject = (s_sub_text, s_mention)
                        o_mention = text[s_obj_text:e_obj_text]
                        pred_object = (s_obj_text, o_mention)
                        pred_spoes.append(
                            (pred_subject, pred_predicate, pred_object))
                #  logger.info(f"pred_spoes: {pred_spoes}")

                self.eval_results.append(spoes)
            x, y, z = evaluate_spo_list(pred_spoes, spo_list)
            X += x
            Y += y
            Z += z
            #  else:
            #      logger.warning(f"subjects is None.")

        eval_loss = eval_loss / num_eval_steps
        results = {}
        #  results = {f'{key}': value for key, value in eval_info.items()}
        results['loss'] = eval_loss
        f1, acc, recall = 2 * X / (Y + Z), X / Y, X / Z
        self.X = X
        self.Y = Y
        self.Z = Z
        results['f1'] = f1
        results['acc'] = acc
        results['recall'] = recall

        return (eval_loss, ), results

    #  def on_eval_step(self, args, eval_dataset, step, model, inputs, outputs):
    #  def on_eval_step_temp(self, args, model, step, batch, batch_features):
    #      all_input_ids, all_input_mask, all_segment_ids, all_sub_labels, all_seed_subs, all_obj_labels, all_lens = batch
    #
    #      eval_loss = 0.0
    #      num_eval_steps = 0
    #      for i in range(all_input_ids.size()[0]):
    #          inputs = {
    #              "input_ids": all_input_ids[i].view(1, -1),
    #              "attention_mask": all_input_mask[i].view(1, -1),
    #          }
    #          if args.model_type != "distilbert":
    #              # XLM and RoBERTa don"t use segment_ids
    #              inputs["token_type_ids"] = (all_segment_ids[i].view(
    #                  1, -1) if args.model_type in ["bert", "xlnet"] else None)
    #
    #          outputs = model(**inputs)
    #          tmp_eval_loss, logits = outputs[:2]
    #          eval_loss += tmp_eval_loss
    #          num_eval_steps += 1
    #
    #          #  loss_sig = nn.Sigmoid()
    #          #  active_logits = logits.view(-1, args.num_labels * 2)
    #          #  activae_logits = loss_sig(active_logits)
    #          #  active_logits = active_logits**2
    #          #  preds = active_logits
    #          preds = get_active_logits(logits, args.num_labels)
    #
    #          T = batch_features[i].subjects
    #          R = extract_entity_from_logits(
    #              args,
    #              preds,
    #              all_lens[i:i + 1],
    #              confidence=args.confidence,
    #              enable_nested_entities=args.enable_nested_entities)
    #          #  logger.info(f"R: {R}")
    #          #  logger.info(f"T: {T}")
    #
    #          self.metric.update(true_subject=T, pred_subject=R)
    #
    #      eval_loss = eval_loss / num_eval_steps
    #      eval_info, entity_info = self.metric.result()
    #      results = {f'{key}': value for key, value in eval_info.items()}
    #      results['loss'] = eval_loss
    #
    #      return (eval_loss, ), results

    def on_predict_start(self, args, test_dataset):
        self.pred_results = []

    def on_predict_step(self, args, model, step, batch):
        all_input_ids, all_input_mask, all_segment_ids, all_sub_labels, all_seed_subs, all_obj_labels, all_lens, all_token_offsets, all_texts, all_spo_lists = batch

        for i in range(all_input_ids.size()[0]):
            inputs = {
                "input_ids": all_input_ids[i].view(1, -1),
                "attention_mask": all_input_mask[i].view(1, -1),
                #  "sub_labels": all_sub_labels[i].view(1, -1),
                #  "seed_subs": all_seed_subs[i].view(1, -1),
                #  "obj_labels": all_obj_labels[i].view(1, -1),
                "do_forward_obj": False,
            }
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (all_segment_ids[i].view(
                    1, -1) if args.model_type in ["bert", "xlnet"] else None)
            token_offsets = all_token_offsets[i]
            text = all_texts[i]
            spo_list = all_spo_lists[i]

            outputs = model(**inputs)
            #  logger.warning(f"ouptuts: {outputs}")

            batch_eval_loss, subject_preds = outputs[:2]

            #  logger.info(f"subject_preds.shape: {subject_preds.shape}")
            subjects = extract_subjects_from_logits(args,
                                                    subject_preds,
                                                    confidence=args.confidence)
            #  logger.warning(f"text: {text}")
            #  logger.info(f"spo_list: {spo_list}")
            pred_subjects = []
            for sub in subjects:
                s_text = token_offsets[sub[0] - 1][0]
                e_text = token_offsets[sub[1] - 1][-1]
                pred_subjects.append(text[s_text:e_text])
            #  logger.info(f"pred_subjects: {pred_subjects}")
            pred_spoes = []
            if subjects:
                spoes = extract_predicate_objects_from_logits(
                    args,
                    model,
                    subjects,
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    inputs['token_type_ids'],
                    token_offsets,
                    all_lens[i:i + 1],
                    confidence=args.confidence / 4)
                #  logger.info(f"Eval spoes: {spoes}")
                for spo in spoes:
                    (s_sub, e_sub), predicate, (s_obj, e_obj) = spo
                    s_sub_text = token_offsets[s_sub - 1][0].item()
                    e_sub_text = token_offsets[e_sub - 1][-1].item()
                    s_obj_text = token_offsets[s_obj - 1][0].item()
                    e_obj_text = token_offsets[e_obj - 1][-1].item()
                    if e_sub_text - s_sub_text >= 2 and e_obj_text - s_obj_text >= 2:
                        #  logger.info(f"{text[s_sub_text:e_sub_text]} , {args.id2label[predicate]}, {text[s_obj_text:e_obj_text]}")
                        pred_predicate = args.id2label[predicate]
                        s_mention = text[s_sub_text:e_sub_text]
                        pred_subject = (s_sub_text, s_mention)
                        o_mention = text[s_obj_text:e_obj_text]
                        pred_object = (s_obj_text, o_mention)
                        pred_spoes.append(
                            (pred_subject, pred_predicate, pred_object))
                #  logger.info(f"pred_spoes: {pred_spoes}")

            self.pred_results.append(pred_spoes)

    def on_predict_end(self, args, test_dataset):
        return self.pred_results

    def on_eval_end(self, args, eval_dataset):
        X = self.X
        Y = self.Y
        Z = self.Z
        f1, acc, recall = 2 * X / (Y + Z), X / Y, X / Z
        results = {}
        results['f1'] = f1
        results['acc'] = acc
        results['recall'] = recall
        return results
