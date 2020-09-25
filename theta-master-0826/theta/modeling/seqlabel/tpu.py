# -*- coding: utf-8 -*-


#%% ------------------------------------------------------------
# TPU
def tpu_available():
    import os
    return 'COLAB_TPU_ADDR' in os.environ


def init_tpu():
    strategy = None
    if tpu_available():
        print(f"TPU available.")
        tf.logging.set_verbosity(tf.logging.INFO)
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_host(resolver.master())
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    return strategy


def convert_data_to_tensors(bert_data):
    X_data = []
    num_steps = len(bert_data)
    print(f"{num_steps} steps.")
    i = 0
    with tqdm(total=num_steps) as pbar:
        for (X1, X2, S1, S2), _ in bert_data.__iter__(max_length):
            if i >= num_steps:
                break
            for x1, x2, s1, s2 in zip(X1, X2, S1, S2):
                X_data.append([x1, x2, s1, s2])
            pbar.update(1)
            i += 1
    return X_data


def train_model_on_tpu(train_data, dev_data):
    print(f"TPU available.")
    batch_size = 128
    tpu_cores = 8
    bert_train_data = BertTrainData(train_data,
                                    max_length,
                                    batch_size=batch_size)
    bert_dev_data = BertTrainData(dev_data,
                                  extract_max_length,
                                  batch_size=batch_size)

    X_train = convert_data_to_tensors(bert_train_data)
    print(f"\nX_train len: {len(X_train)}")
    X_dev = convert_data_to_tensors(bert_dev_data)
    print(f"\nX_dev {len(X_dev)}")

    X_train = []
    num_steps = len(bert_train_data)
    print(f"{num_steps} steps.")
    i = 0
    with tqdm(total=num_steps) as pbar:
        for (X1, X2, S1, S2), _ in bert_train_data.__iter__(max_length):
            if i >= num_steps:
                break
            #print(f"X1.shape: {X1.shape} X2.shape: {X2.shape} S1.shape: {S1.shape} S2.shape: {S2.shape}")
            #X = np.concatenate([X1, X2, S1, S2], axis=1).reshape((-1, 4, X1.shape[1]))
            #X_train.append(list(X))
            for x1, x2, s1, s2 in zip(X1, X2, S1, S2):
                X_train.append([x1, x2, s1, s2])
            pbar.update(1)
            i += 1

    print(f"\nX_train.shape: {len(X_train)}")

    print(f"batch_size: {batch_size} tpu_cores: {tpu_cores}")
    strategy = init_tpu()
    with strategy.scope():
        X_train = convert_data_to_tensors(bert_train_data)
        print(f"\nX_train len: {len(X_train)}")
        X_dev = convert_data_to_tensors(bert_dev_data)
        print(f"\nX_dev {len(X_dev)}")
        train_model.fit(
            X_train,
            steps_per_epoch=len(bert_train_data) / tpu_cores,
            #steps_per_epoch=int(len(bert_train_data) / (batch_size * tpu_cores)),
            epochs=8,
            batch_size=batch_size * 8,
            validation_data=(X_dev, 0))


#  X_train = convert_data_to_tensors(bert_train_data)
#  print(f"\nX_train len: {len(X_train)}")
#  X_dev = convert_data_to_tensors(bert_dev_data)
#  print(f"\nX_dev {len(X_dev)}")
#  bert_model.train_model.fit(
#      X_train,
#      #  steps_per_epoch=len(bert_train_data) / tpu_cores,
#      #steps_per_epoch=int(len(bert_train_data) / (batch_size * tpu_cores)),
#      epochs=3,
#      batch_size=batch_size * gpus,
#      callbacks=[evaluator])

#  if tpu_available():
#      self.on_epoch_begin_tpu(batch, logs)
#      return

#  def get_lr_opt(self):
#      if not hasattr(self, 'model'):
#          return None, None
#      tmp = self.model
#      lr_attr_name = None
#      lr_name_list = ['lr', '_lr', '_learning_rate', 'learning_rate']
#      opt_name_list = ['optimizer', '_opt']
#      while True:
#          for lr_name in lr_name_list:
#              if hasattr(tmp, lr_name):
#                  lr_attr_name = lr_name
#                  break
#          if lr_attr_name != None:
#              return tmp, lr_attr_name
#          for opt_name in opt_name_list:
#              if hasattr(tmp, opt_name):
#                  tmp = getattr(tmp, opt_name)
#                  break
#      return None, None
#
#  def on_epoch_begin_tpu(self, epoch, logs=None):
#      opt, lr_name = self.get_lr_opt()
#      if opt is None:
#          raise ValueError('no attr _lr, _learning or lr found')
#      try:  # new API
#          lr = getattr(opt, lr_name)
#          lr = self.schedule(epoch, lr)
#      except TypeError:  # Support for old API for backward compatibility
#          lr = self.schedule(epoch)
#      if not isinstance(lr, (float, np.float32, np.float64)):
#          raise ValueError('The output of the "schedule" function '
#                           'should be float.')
#      setattr(opt, lr_name, lr)
#      if self.verbose > 0:
#          print('\nEpoch %05d: LearningRateScheduler reducing learning '
#                'rate to %s.' % (epoch + 1, lr))
