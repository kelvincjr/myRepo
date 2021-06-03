import os
import torch

data_dir = os.getcwd() + '/data/clue/'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'test']
#bert_model = '/opt/kelvin/python/knowledge_graph/ai_contest/working'
#roberta_model = '/opt/kelvin/python/knowledge_graph/ai_contest/working'
bert_model = '/kaggle/working'
roberta_model = '/kaggle/working'
model_dir = os.getcwd() + '/experiments/commodity/'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

#batch_size = 32
#epoch_num = 50
batch_size = 8
epoch_num = 20
min_epoch_num = 5
patience = 0.0002
patience_num = 10

gpu = '0'

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

'''
labels = ['address', 'book', 'company', 'game', 'government',
          'movie', 'name', 'organization', 'position', 'scene']

label2id = {
    "O": 0,
    "B-address": 1,
    "B-book": 2,
    "B-company": 3,
    'B-game': 4,
    'B-government': 5,
    'B-movie': 6,
    'B-name': 7,
    'B-organization': 8,
    'B-position': 9,
    'B-scene': 10,
    "I-address": 11,
    "I-book": 12,
    "I-company": 13,
    'I-game': 14,
    'I-government': 15,
    'I-movie': 16,
    'I-name': 17,
    'I-organization': 18,
    'I-position': 19,
    'I-scene': 20,
    "S-address": 21,
    "S-book": 22,
    "S-company": 23,
    'S-game': 24,
    'S-government': 25,
    'S-movie': 26,
    'S-name': 27,
    'S-organization': 28,
    'S-position': 29,
    'S-scene': 30
}

id2label = {_id: _label for _label, _id in list(label2id.items())}
'''
train_word_file = r'./commodity_data/train_all/input.seq.char'
train_label_file = r'./commodity_data/train_all/output.seq.bioattr'
test_word_file = r'./commodity_data/test_all/input.seq.char'
test_label_file = r'./commodity_data/test_all/output.seq.bioattr'

label_file = r'./commodity_data/vocab_attr.txt'
label_bio_file = r'./commodity_data/vocab_bioattr.txt'
vocab_file = r'./commodity_data/vocab_char.txt'

def read_labels(filename):
    # 读取schema
    with open(filename, encoding='utf-8') as f:
        labels = []
        for l in f:
            if l.strip() != 'null':
                labels.append(l.strip())
        return labels

def read_label2id(filename):
    # 读取schema
    with open(filename, encoding='utf-8') as f:
        label2id = dict()
        id2label = dict()
        count = 0
        for l in f:
            token = l.strip()
            label2id[token] = count
            id2label[count] = token
            count += 1
        return label2id, id2label

def load_vocabulary(path):

    print(" load start")

    w2i = {}
    i2w = {}
    index = 0

    with open(path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            word = line[:-1]
            w2i[word] = index
            i2w[index] = word
            index += 1

    print("vocab from: {}, containing words: {}".format(path, len(i2w)))

    return w2i, i2w

labels = read_labels(label_file)
label2id, id2label = read_label2id(label_bio_file)

w2i_char, i2w_char = load_vocabulary(vocab_file)
