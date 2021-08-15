# Theta 

Deep learning toolbox for end-to-end text information extraction tasks.

Theta定位是解决实际工程项目中文本信息抽取任务的实用工具箱，端到端实现从原始文本输入到结构化输出全过程。用户工作聚焦于输入数据格式转换，调整关键参数调度theta完成模型训练推理任务及输出格式化数据利用。

Theta应用场景包括国家级重点企业非结构化数据挖掘利用、开放域文本数据结构化抽取、各大在线实体关系抽取类评测赛事等。

Theta性能指标要求达到业内主流头部水准，近期参加了包括CCF2019、CHIP2019、CCKS2020、CCL2020等C字头顶级赛事，目前取得10余次决赛奖项，包括7次前三，2次第一。


## 更新

2020.07.03 - 版本 0.22.0

经过近期十几场线上比赛评测实战，theta开始趋于稳定。以版本0.22.0为起点，theta的开发工作将转入独立的框架层面，应用开发都将基于pip安装版本。辅助调试theta的几场线上比赛作为theta应用示例放在`examples/CCKS2000`目录下。

- [CCKS 2020：面向金融领域的篇章级事件主体与要素抽取（二）篇章事件要素抽取](https://www.biendata.xyz/competition/ccks_2020_4_2/)，线上得分0.74917，目前排名第2。

- [CCKS 2020：面向中文电子病历的医疗实体及事件抽取（二）医疗事件抽取](https://www.biendata.xyz/competition/ccks_2020_2_1/)，线上得分0.74125，目前排名第4。

- [CCKS 2020：面向中文电子病历的医疗实体及事件抽取（一）医疗命名实体识别](https://www.biendata.xyz/competition/ccks_2020_2_2/)，线上成绩0.85663，目前排名第6。

  

2020.06.19 - 版本 0.21.0

- 新增文本分类任务案例：CCKS 2020：新冠知识图谱构建与问答评测（一）新冠百科知识图谱类型推断(examples/CCKS2020/entity_typing)，线上成绩0.93720，目前排名第1。
- 新增mlflow训练跟踪，记录每次实验的参数、评估指标等。

## 安装

测试版
```
pip install git+http://122.112.206.124:3000/idleuncle/theta@0.22.0
```
正式版

```
pip install theta==0.22.0
```
## CCKS 2020：新冠知识图谱构建与问答评测（一）新冠百科知识图谱类型推断

[赛事网址](https://www.biendata.xyz/competition/ccks_2020_7_1/) 线上成绩0.93720

完整代码见theta/examples/CCKS2020/entity_typing/

本评测任务围绕新冠百科知识图谱构建中的实体类型推断（Entity Type Inference）展开。评测从实体百科（包括百度百科、互动百科、维基百科、医学百科）页面出发，从给定的数据中推断相关实体的类型。

输入样例 

entity.txt:

> 烟草花叶病毒
> 大肠杆菌
> 艾滋病
> 盐酸西普利嗪
> 内科
> 太阳

type.txt：

> 病毒
> 细菌
> 疾病
> 药物
> 医学专科
> 检查科目
> 症状

**输出样例**

> 烟草花叶病毒    病毒
> 大肠杆菌    细菌
> 艾滋病    疾病
> 盐酸西普利嗪    药物
> 内科    医学专科
> 太阳    NoneType




## CLUE-CLUENER 细粒度命名实体识别

本数据是在清华大学开源的文本分类数据集THUCTC基础上，选出部分数据进行细粒度命名实体标注，原数据来源于Sina News RSS.

训练集：10748 验证集：1343

标签类别： 数据分为10个标签类别，分别为: 地址（address），书名（book），公司（company），游戏（game），政府（goverment），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）

数据下载地址：https://github.com/CLUEbenchmark/CLUENER2020

排行榜地址：https://cluebenchmarks.com/ner.html

完整代码见theta/examples/CLUENER：[cluener.ipynb](theta/examples/CLUENER/cluener.ipynb)

选用bert-base-chinese预训练模型，CLUE测评F1得分77.160。

```
# 训练
make -f Makefile.cluener train

# 推理
make -f Makefile.cluener predict

# 生成提交结果文件
make -f Makefile.cluener submission
```

## CLUE-TNEWS 今日头条中文新闻（短文）分类任务

以下样例是CLUE（[中文任务基准测评](https://cluebenchmarks.com/index.html)）中今日头条中文新闻（短文）分类任务。

数据集来自今日头条的新闻版块，共提取了15个类别的新闻，包括旅游，教育，金融，军事等。

数据量：训练集(53,360)，验证集(10,000)，测试集(10,000)

> 例子：
> {"label": "102", "label_desc": "news_entertainment", "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物"}
> 每一条数据有三个属性，从前往后分别是 分类ID，分类名称，新闻字符串（仅含标题）。

选用bert-base-chinese预训练模型，CLUE测评F1得分56.100。

完整代码见theta/examples/TNEWS：[tnews.ipynb](theta/examples/TNEWS/tnews.ipynb)

[TNEWS数据集下载](https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip)

### 导入基础库

```
import json
from tqdm import tqdm
from loguru import logger
import numpy as np

from theta.modeling import load_glue_examples
from theta.modeling.glue import GlueTrainer, load_model, get_args
from theta.utils import load_json_file
```

### 自定义数据生成器

多数情况下，此节是唯一需要自行修改的部分。

根据实际需要处理数据的格式，自定义样本标签集和训练数据、验证数据、测试数据生成器。

样本标签集glue_labels定义分类标签列表，列表项是实际需要输出的标签字符串。

数据生成器需要遵守的基本规范是生成器逐行返回(guid, text_a, text_b, label)四元组。


```
# 样本标签集
labels_file = './data/labels.json'
labels_data = load_json_file(labels_file)
glue_labels = [x['label_desc'] for x in labels_data]
logger.info(f"glue_labels: {len(glue_labels)} {glue_labels}")


def clean_text(text):
    text = text.strip().replace('\n', '')
    text = text.replace('\t', ' ')
    return text


# 训练数据生成器
def train_data_generator(train_file):
    train_data = load_json_file(train_file)
    for i, json_data in enumerate(tqdm(train_data, desc="train")):
        guid = str(i)
        text = json_data['sentence']
        text = clean_text(text)
        label = json_data['label_desc']

        yield guid, text, None, label


# 验证数据生成器
def eval_data_generator(eval_file):
    eval_data = load_json_file(eval_file)
    for i, json_data in enumerate(tqdm(eval_data, desc="eval")):
        guid = str(i)
        text = json_data['sentence']
        text = clean_text(text)
        label = json_data['label_desc']

        yield guid, text, None, label


# 测试数据生成器
def test_data_generator(test_file):
    test_data = load_json_file(test_file)
    total_examples = len(test_data)
    for i, json_data in enumerate(tqdm(test_data, desc="test")):
        guid = str(json_data['id'])
        text = json_data['sentence']
        text = clean_text(text)

        yield guid, text, None, None
```

### 载入数据集

以下代码不需要修改，原样使用即可。

```
# 载入训练数据集
def load_train_examples(train_file):
    train_examples = load_glue_examples(train_data_generator, train_file)
    logger.info(f"Loaded {len(train_examples)} train examples.")

    return train_examples


# 载入验证数据集
def load_eval_examples(eval_file):
    eval_examples = load_glue_examples(eval_data_generator, eval_file)
    logger.info(f"Loaded {len(eval_examples)} eval examples.")

    return eval_examples


# 载入测试数据集
def load_test_examples(test_file):
    test_examples = load_glue_examples(test_data_generator, test_file)
    logger.info(f"Loaded {len(test_examples)} test examples.")

    return test_examples
```

### 自定义模型

Theta提供缺省模型，多数情况下不需要自定义模型。关于自定义模型的详细情况，在进阶文档中说明。

### 自定义训练器

当使用缺省模型时，训练器也是不需要定义的，直接使用AppTrainer=GlueTrainer即可。

通常自定义训练器的目的是通过重载Trainer，获取训练及推理过程的实时数据。

```
class AppTrainer(GlueTrainer):
    def __init__(self, args, glue_labels):
        # 使用自定义模型时，传入build_model参数。
        super(AppTrainer, self).__init__(args, glue_labels, build_model=None)
```

### 主函数

主函数是固定套路，通常不需要修改。 
可以在add_special_args函数中定义自行需要的命令行参数，并在main函数中处理，具体例子见以下do_eda参数。

```
def main(args):

    if args.do_eda:
        from theta.modeling import show_glue_datainfo
        show_glue_datainfo(glue_labels, train_data_generator, args.train_file,
                           test_data_generator, args.test_file)
    else:
        trainer = AppTrainer(args, glue_labels)

        # --------------- Train ---------------
        if args.do_train:
            train_examples = load_train_examples(args.train_file)
            eval_examples = load_eval_examples(args.eval_file)
            trainer.train(args, train_examples, eval_examples)

        # --------------- Evaluate ---------------
        elif args.do_eval:
            eval_examples = load_eval_examples(args.eval_file)
            model = load_model(args)
            trainer.evaluate(args, model, eval_examples)

        # --------------- Predict ---------------
        elif args.do_predict:
            test_examples = load_test_examples(args)
            model = load_model(args)
            trainer.predict(args, model, test_examples)

            save_predict_results(args, trainer.pred_results,
                                 f"./{args.dataset_name}_predict.json",
                                 test_examples)


if __name__ == '__main__':

    def add_special_args(parser):
        parser.add_argument("--do_eda", action="store_true")
        return parser

    from theta.modeling.glue import get_args
    args = get_args([add_special_args])
    main(args)

```
### 启动训练

```
	python run_tnews.py \
		--do_train \
		--model_type bert \
		--model_path /opt/share/pretrained/pytorch/bert-base-chinese 
		--data_dir ./data \
		--output_dir ./output \
		--dataset_name tnews \
		--train_file ./data/train.json
		--learning_rate 2e-5 \
		--train_max_seq_length 160 \
		--per_gpu_train_batch_size 64 \
		--per_gpu_eval_batch_size 64 \
		--num_train_epochs 10 
```

### 启动验证

```
	python run_tnews.py \
		--do_eval \
		--model_type bert \
		--model_path ./output/best \
		--data_dir ./data \
		--output_dir ./output \
		--dataset_name tnews \
		--eval_file ./data/dev.json
		--eval_max_seq_length 160 \
		--per_gpu_eval_batch_size 64 
```

### 启动推理
```
	python run_tnews.py \
		--do_predict \
		--model_type bert \
		--model_path ./output/best \
		--data_dir ./data \
		--output_dir ./output \
		--dataset_name tnews \
		--test_file ./data/test.json
		--eval_max_seq_length 160 \
		--per_gpu_predict_batch_size 64 
```
