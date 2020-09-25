# -*- coding: utf-8 -*-
import os, re
import numpy as np
from loguru import logger

#  import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Lambda, Input
import keras.backend as K
from keras.optimizers import Adam, RMSprop
from adamw import AdamW
from keras.utils import multi_gpu_model

#  os.environ.setdefault('TF_KERAS', '1')
print(f"TF_KERAS={os.environ.get('TF_KERAS', '0')}")
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

from train_config import Config

# for keras_bert.load_trained_model_from_checkpoint()
#  bert_seq_len = 512
#  extract_max_length = Config.max_seq_length + 2  # bert_seq_len  # - 2  #180  #1024

debug_extract_labels = Config.debug_extract_labels


#%% ------------------------------------------------------------
# Model
class BertTokenizer(Tokenizer):
    def __init__(self, bert_vocab_file: str):
        tokens = {}
        for line in open(bert_vocab_file, 'r').readlines():
            token = line.strip()
            tokens[token] = len(tokens)
        Tokenizer.__init__(self, tokens)

    def _tokenize(self, text):
        return [
            x if x in self._token_dict else
            # space类用未经训练的[unused1]表示，
            # 未在字典中的字符用[UNK]表示。
            ('[unused1]' if self._is_space(x) else '[UNK]') for x in text
        ]


def load_bert_pretrained_model(max_seq_length):
    extract_max_length = max_seq_length + 2
    bert_pretrained_model = load_trained_model_from_checkpoint(
        Config.config_path, Config.checkpoint_path, seq_len=extract_max_length)
    if Config.gpus > 1:
        bert_pretrained_model = multi_gpu_model(bert_pretrained_model,
                                                gpus=Config.gpus)
    for layer in bert_pretrained_model.layers:
        layer.trainable = True

    return bert_pretrained_model


def find_pos_pairs(ps1, ps2, s_threshold=0.01, e_threshold=0.01, top_k=0):
    assert len(ps1) == len(ps2)
    s = [(prob, i) for i, prob in enumerate(ps1)]
    e = [(prob, i) for i, prob in enumerate(ps2)]
    s = sorted(s,
               key=lambda x: f"{x[0]:.4f},{1 /(x[1] + 1):.6f}",
               reverse=True)
    e = sorted(e,
               key=lambda x: f"{x[0]:.4f},{1 /(x[1] + 1):.6f}",
               reverse=True)
    #  logger.debug(f"s: {s}")
    #  logger.debug(f"e: {e}")
    s = [x for x in s if x[0] >= s_threshold]
    e = [x for x in e if x[0] >= e_threshold]
    if top_k > 0:
        s = s[:top_k]
        e = e[:top_k]

    #  logger.debug(f"filtered s: {s}")
    #  logger.debug(f"filtered e: {e}")

    return s, e


def find_entities_from_indices(indices, min_label_len):
    dangling_entity = None
    s_idx = -1
    for s_idx, (_, _, (tag, idx, prob, _)) in enumerate(indices):
        if tag == 's':
            break
    pairs = []
    if s_idx >= 0:
        text_in_0, c_in_0, (tag0, idx0, prob0, text_offset) = indices[s_idx]
        idx0 += text_offset
        indices = indices[s_idx + 1:]
        prob1 = 0.0
        for text_in, c_in, (tag, idx, prob, text_offset) in indices:
            idx += text_offset

            if tag == tag0:
                if tag0 == 's':
                    if prob > prob0:
                        idx0 = idx
                        prob0 = prob
                        text_in_0 = text_in
                        c_in_0 = c_in
                else:
                    if prob > prob1:
                        idx1 = idx
                        prob1 = prob
                        pairs[-1] = (pairs[-1][0], (idx1, prob1), text_in,
                                     c_in)
            else:
                if tag0 == 's':
                    # 's' -> 'e'
                    idx1 = idx
                    prob1 = prob
                    pairs.append(((idx0, prob0), (idx1, prob1), text_in, c_in))
                else:
                    # 'e' -> 's'
                    idx0 = idx
                    prob0 = prob
                    text_in_0 = text_in
                    c_in_0 = c_in
                tag0 = tag
                idx0 = idx
                text_in_0 = text_in
                c_in_0 = c_in
        if tag0 == 's':
            #  dangling_entity = (idx0 if idx0 == 0 else idx0 - 1, prob0,
            dangling_entity = (idx0, prob0, text_in_0, c_in_0)
    #  pairs = [((idx0 if idx0 == 0 else idx0 - 1, prob0), (idx1, prob1), text_in,

    pairs = [((idx0, prob0), (idx1, prob1), text_in, c_in)
             for (idx0, prob0), (idx1, prob1), text_in, c_in in pairs
             if idx1 - idx0 >= min_label_len]

    entities = [(idx0, idx1, prob0, prob1, text_in, c_in)
                for (idx0, prob0), (idx1, prob1), text_in, c_in in pairs]

    return entities, dangling_entity


class BertModel():
    def __init__(self, config_path, checkpoint_path, dict_path, learning_rate,
                 max_seq_length):

        self.tokenizer = BertTokenizer(dict_path)
        self.learning_rate = learning_rate
        self.x1_in = Input(shape=(None, ))  # 待识别句子输入
        self.x2_in = Input(shape=(None, ))  # 待识别句子输入
        self.s1_in = Input(shape=(None, ))  # 实体左边界（标签）
        self.s2_in = Input(shape=(None, ))  # 实体右边界（标签）

        #  bert_pretrained_model = load_trained_model_from_checkpoint(
        #      config_path, checkpoint_path, seq_len=extract_max_length)
        #  if Config.gpus > 1:
        #      bert_pretrained_model = multi_gpu_model(bert_pretrained_model,
        #                                              gpus=Config.gpus)
        #
        #  for layer in bert_pretrained_model.layers:
        #      layer.trainable = True
        bert_pretrained_model = load_bert_pretrained_model(max_seq_length)

        self.bert_pretrained_model = bert_pretrained_model  #to_tpu_model(bert_pretrained_model)

    def build_model(self):
        x1, x2, s1, s2 = self.x1_in, self.x2_in, self.s1_in, self.s2_in
        self.x_mask = Lambda(
            lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)

        x = self.bert_pretrained_model([x1, x2])

        # BiGRU + DNN
        #  # from https://github.com/hecongqing/CCKS2019EventEntityExtraction_Rank5/blob/master/src/SEBERT_model.py
        #  l = Lambda(lambda t: t[:, -1])(x)
        #  x = Add()([x, l])
        #  x = Dropout(0.1)(x)
        #  x = Lambda(lambda x: x[0] * x[1])([x, x_mask])
        #
        #  x = SpatialDropout1D(0.1)(x)
        #  x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
        #  x = Lambda(lambda x: x[0] * x[1])([x, x_mask])
        #  x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
        #  x = Lambda(lambda x: x[0] * x[1])([x, x_mask])
        #
        #  x = Dense(1024, use_bias=False, activation='tanh')(x)
        #  x = Dropout(0.2)(x)
        #  x = Dense(64, use_bias=False, activation='tanh')(x)
        #  x = Dropout(0.2)(x)
        #  x = Dense(8, use_bias=False, activation='tanh')(x)

        ps1 = Dense(1, use_bias=False)(x)
        ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)(
            [ps1, self.x_mask])
        ps2 = Dense(1, use_bias=False)(x)
        ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)(
            [ps2, self.x_mask])

        self.predict_model = Model([self.x1_in, self.x2_in], [ps1, ps2])

        train_model = Model([self.x1_in, self.x2_in, self.s1_in, self.s2_in],
                            [ps1, ps2])
        if Config.gpus > 1:
            train_model = multi_gpu_model(train_model, gpus=Config.gpus)

        def get_loss(y_true, y_pred, with_weights=False):
            weights = 0.0
            if with_weights:
                # 根据标签位置距离，计算分类权重。
                i_true = K.argmax(y_true, axis=1)
                i_pred = K.argmax(y_pred, axis=1)
                distance = K.abs(i_true - i_pred)
                weights = K.cast(distance, dtype='float32')
                #  length = K.int_shape(y_true)[1] - 1
                #  weights = K.cast(distance / length, dtype='float32')

            losses = (1.0 + weights) * K.categorical_crossentropy(
                y_true, y_pred, from_logits=True)
            #  losses = (
            #      (1.0 + weights) *
            #      K.categorical_crossentropy(y_true, y_pred, from_logits=True))
            loss = K.mean(losses)
            return loss

        #  loss1 = K.mean(
        #      K.categorical_crossentropy(self.s1_in, ps1, from_logits=True))
        #  ps2 -= (1 - K.cumsum(s1, 1)) * 1e10
        #  loss2 = K.mean(
        #      K.categorical_crossentropy(self.s2_in, ps2, from_logits=True))
        #  self.loss = loss1 + loss2
        loss1 = get_loss(self.s1_in, ps1, with_weights=True)
        ps2 -= (1 - K.cumsum(s1, 1)) * 1e10
        loss2 = get_loss(self.s2_in, ps2, with_weights=True)
        self.loss = loss1 + loss2

        train_model.add_loss(self.loss)

        if 'COLAB_TPU_ADDR' in os.environ:
            train_model.compile(
                #optimizer=tf.train.RMSPropOptimizer(self.learning_rate))
                optimizer=RMSprop())
        else:
            #  from accum_optimizer import AccumOptimizer
            #  train_model.compile(optimizer=AccumOptimizer(
            #      Adam(self.learning_rate), steps_per_update))
            train_model.compile(optimizer=AdamW(self.learning_rate))
        train_model.summary()

        self.train_model = train_model

    def determine_entity_positions(self,
                                   text_in,
                                   c_in,
                                   text_offset,
                                   top_k=20,
                                   s_threshold=0.01,
                                   e_threshold=0.01,
                                   seg_len=510,
                                   seg_backoff=400):
        def softmax(x):
            x = x - np.max(x)
            x = np.exp(x)
            return x / np.sum(x)

        text_prefix = f"___{c_in}___"
        text_in = f"{text_prefix}{text_in}"
        #  text_in = u'___%s___%s' % (c_in, text_in)
        text_in = text_in[:seg_len]
        #  text_in += " " * (seg_len- len(text_in))

        _tokens = self.tokenizer.tokenize(text_in)
        num_tokens = len(_tokens)
        _x1, _x2 = self.tokenizer.encode(first=text_in)
        _x1, _x2 = np.array([_x1]), np.array([_x2])
        #  print(f"_x1.shape: {_x1.shape} _x2.shape: {_x2.shape}")
        #  print(f"{len(text_in)} : {text_in[:30]}")
        _ps1, _ps2 = self.predict_model.predict([_x1, _x2])
        #  print(f"_ps1: {_ps1}")
        #  print(f"_ps2: {_ps2}")
        _ps1, _ps2 = softmax(_ps1[0]), softmax(_ps2[0])
        #  print(f"-------- after softmax --------")
        #  print(f"_ps1: {_ps1}")
        #  print(f"_ps2: {_ps2}")

        s_indices, e_indices = find_pos_pairs(_ps1,
                                              _ps2,
                                              s_threshold=s_threshold,
                                              e_threshold=e_threshold,
                                              top_k=top_k)
        #  print("s_indices:", s_indices)
        #  print("e_indices:", e_indices)
        indices = [('s', idx - len(text_prefix) - 1, prob, text_offset)
                   for prob, idx in s_indices
                   ] + [('e', idx - len(text_prefix), prob, text_offset)
                        for prob, idx in e_indices]
        indices = sorted(indices, key=lambda x: x[1])
        if debug_extract_labels:
            logger.debug(f"indices: {indices}")

        return indices

    #  def extract_entity(self, text_in, c_in, additional_chars):
    #      def softmax(x):
    #          x = x - np.max(x)
    #          x = np.exp(x)
    #          return x / np.sum(x)
    #
    #      text_in = u'___%s___%s' % (c_in, text_in)
    #      text_in = text_in[:Config.extract_max_length]
    #      #  text_in += " " * (Config.extract_max_length - len(text_in))
    #
    #      _tokens = self.tokenizer.tokenize(text_in)
    #      num_tokens = len(_tokens)
    #      _x1, _x2 = self.tokenizer.encode(first=text_in)
    #      _x1, _x2 = np.array([_x1]), np.array([_x2])
    #      #  print(f"_x1.shape: {_x1.shape} _x2.shape: {_x2.shape}")
    #      #  print(f"{len(text_in)} : {text_in[:30]}")
    #      _ps1, _ps2 = self.predict_model.predict([_x1, _x2])
    #      #  print(f"_ps1: {_ps1}")
    #      #  print(f"_ps2: {_ps2}")
    #      _ps1, _ps2 = softmax(_ps1[0]), softmax(_ps2[0])
    #      #  print(f"-------- after softmax --------")
    #      #  print(f"_ps1: {_ps1}")
    #      #  print(f"_ps2: {_ps2}")
    #
    #      s_indices, e_indices = find_pos_pairs(_ps1,
    #                                            _ps2,
    #                                            s_threshold=0.1,
    #                                            e_threshold=0.01,
    #                                            top_k=20)
    #
    #      #  print("s_indices:", s_indices)
    #      #  print("e_indices:", e_indices)
    #      indices = [('s', idx, prob)
    #                 for prob, idx in s_indices] + [('e', idx, prob)
    #                                                for prob, idx in e_indices]
    #      indices = sorted(indices, key=lambda x: x[1])
    #      #  print("indices:", indices)
    #
    #      s_idx = -1
    #      for s_idx, (tag, idx, prob) in enumerate(indices):
    #          if tag == 's':
    #              break
    #      pairs = []
    #      if s_idx >= 0:
    #          tag0, idx0, prob0 = indices[s_idx]
    #          indices = indices[s_idx + 1:]
    #          for tag, idx, prob in indices:
    #              if tag == tag0:
    #                  if prob > prob0:
    #                      if tag0 == 's':
    #                          idx0 = idx
    #                          prob0 = prob
    #                      else:
    #                          idx1 = idx
    #                          prob1 = prob
    #                          pairs[-1] = (pairs[-1][0], (idx1, prob1))
    #              else:
    #                  if tag0 == 's':
    #                      # 's' -> 'e'
    #                      idx1 = idx
    #                      prob1 = prob
    #                      pairs.append(((idx0, prob0), (idx1, prob1)))
    #                  else:
    #                      # 'e' -> 's'
    #                      idx0 = idx
    #                      prob0 = prob
    #                  tag0 = tag
    #                  idx0 = idx
    #                  prob0 = prob
    #
    #      pairs = [((idx0, prob0), (idx1, prob1))
    #               for (idx0, prob0), (idx1, prob1) in pairs if idx1 - idx0 > 3]
    #      #  print("pairs:", pairs)
    #      pairs = [(idx0, idx1) for (idx0, _), (idx1, _) in pairs]
    #      #  for idx0, idx1 in pairs:
    #      #      print(f"-------------------- ({idx0},{idx1}) --------------------")
    #      #      print(text_in[idx0:idx1])
    #      #      print("")
    #
    #      for i, _t in enumerate(_tokens):
    #          if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]',
    #                                         _t) and _t not in additional_chars:
    #              _ps1[i] -= 10
    #      start = _ps1.argmax()
    #      for end in range(start, num_tokens):
    #          _t = _tokens[end]
    #          if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]',
    #                                         _t) and _t not in additional_chars:
    #              break
    #      end = _ps2[start:end + 1].argmax() + start
    #      #  print(f"start: {start}, end: {end}")
    #      a = text_in[start - 1:end]
    #
    #      k1, k2 = np.zeros(num_tokens), np.zeros(num_tokens)
    #      if start >= 0:
    #          k1[start] = 1
    #          k2[end] = 1
    #
    #      return a, (k1.tolist(), k2.tolist())

    #  def extract_entities(self,
    #                       text_in,
    #                       c_in,
    #                       top_k=20,
    #                       s_threshold=0.01,
    #                       e_threshold=0.01,
    #                       min_label_len=8):
    #      def softmax(x):
    #          x = x - np.max(x)
    #          x = np.exp(x)
    #          return x / np.sum(x)
    #
    #      text_in = u'___%s___%s' % (c_in, text_in)
    #      text_in = text_in[:Config.extract_max_length]
    #      #  text_in += " " * (Config.extract_max_length - len(text_in))
    #
    #      dangling_entity = None
    #
    #      _tokens = self.tokenizer.tokenize(text_in)
    #      num_tokens = len(_tokens)
    #      _x1, _x2 = self.tokenizer.encode(first=text_in)
    #      _x1, _x2 = np.array([_x1]), np.array([_x2])
    #      #  print(f"_x1.shape: {_x1.shape} _x2.shape: {_x2.shape}")
    #      #  print(f"{len(text_in)} : {text_in[:30]}")
    #      _ps1, _ps2 = self.predict_model.predict([_x1, _x2])
    #      #  print(f"_ps1: {_ps1}")
    #      #  print(f"_ps2: {_ps2}")
    #      _ps1, _ps2 = softmax(_ps1[0]), softmax(_ps2[0])
    #      #  print(f"-------- after softmax --------")
    #      #  print(f"_ps1: {_ps1}")
    #      #  print(f"_ps2: {_ps2}")
    #
    #      s_indices, e_indices = find_pos_pairs(_ps1,
    #                                            _ps2,
    #                                            s_threshold=s_threshold,
    #                                            e_threshold=e_threshold,
    #                                            top_k=top_k)
    #      #  print("s_indices:", s_indices)
    #      #  print("e_indices:", e_indices)
    #      indices = [('s', idx, prob)
    #                 for prob, idx in s_indices] + [('e', idx, prob)
    #                                                for prob, idx in e_indices]
    #      indices = sorted(indices, key=lambda x: x[1])
    #      #  print("indices:", indices)
    #
    #      s_idx = -1
    #      for s_idx, (tag, idx, prob) in enumerate(indices):
    #          if tag == 's':
    #              break
    #      pairs = []
    #      if s_idx >= 0:
    #          tag0, idx0, prob0 = indices[s_idx]
    #          indices = indices[s_idx + 1:]
    #          for tag, idx, prob in indices:
    #              if tag == tag0:
    #                  if prob > prob0:
    #                      if tag0 == 's':
    #                          idx0 = idx
    #                          prob0 = prob
    #                      else:
    #                          idx1 = idx
    #                          prob1 = prob
    #                          pairs[-1] = (pairs[-1][0], (idx1, prob1))
    #              else:
    #                  if tag0 == 's':
    #                      # 's' -> 'e'
    #                      idx1 = idx
    #                      prob1 = prob
    #                      pairs.append(((idx0, prob0), (idx1, prob1)))
    #                  else:
    #                      # 'e' -> 's'
    #                      idx0 = idx
    #                      prob0 = prob
    #                  tag0 = tag
    #                  idx0 = idx
    #                  prob0 = prob
    #          if tag0 == 's' and idx0 < len(text_in):
    #              dangling_entity = (idx0 if idx0 == 0 else idx0 - 1, prob0,
    #                                 text_in[idx0 if idx0 == 0 else idx0 - 1:-1])
    #      pairs = [((idx0 if idx0 == 0 else idx0 - 1, prob0), (idx1, prob1))
    #               for (idx0, prob0), (idx1, prob1) in pairs
    #               if idx1 - idx0 >= min_label_len]
    #
    #      entities = [(text_in[idx0:idx1], idx0, idx1, prob0, prob1)
    #                  for (idx0, prob0), (idx1, prob1) in pairs]
    #
    #      return entities, dangling_entity
