import json
from tqdm import tqdm
from loguru import logger
import pandas as pd
import numpy as np
from collections import Counter

import sys
sys.path.append('../../../')

seg_len=0
seg_backoff=0
fold = 0

train_labeled_spo_file = r'data\train\labeled.json'
test_spo_file = r'data\train\test.json'
train_unlabeled_spo_file = r'data\train\unlabeled.json'
schema_file = r'data\schema.csv'

def load_spo_data(filename):
    D = []
    num_lines = 0
    with open(filename, encoding='utf-8') as f:
    	#l = f.readlines()
    	data_list = json.load(f)
    	for data in data_list:
    		spo_id = data['id']
    		text = data['text']
    		spo_list = data['spo_list']
    		print('id: {}, text: {}, spo_list: {}'.format(spo_id, text, spo_list))
    		break

load_spo_data(train_labeled_spo_file)


