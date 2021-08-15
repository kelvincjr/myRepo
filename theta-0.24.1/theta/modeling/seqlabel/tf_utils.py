# -*- coding: utf-8 -*-



def enable_gpu_growth():
    pass
    #  import tensorflow as tf
    #  import keras.backend as K
    #  config = tf.ConfigProto()
    #  config.gpu_options.allow_growth = True
    #  sess = tf.Session(config=config)
    #  K.set_session(sess)
    #

def get_gpu_info(gpu_id: int = 0):
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(
        f"GPU {gpu_id} mem: total: {meminfo.total}, used: {meminfo.used}, free: {meminfo.free}"
    )
    return meminfo
