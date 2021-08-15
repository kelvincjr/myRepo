#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
from loguru import logger

from ..base_model import TransformerModel


class EntityExtractionModel(TransformerModel):
    def __init__(self,
                 args,
                 entity_labels,
                 tokenizer,
                 tagging_type="pointer_sequence"):
        """
        实体抽取模型
        按不同的标注方式构建实体抽取模型

        :param entity_labels:List(str) 实体类型标签列表
        :param tokenizer:tokenizers.Tokenizer
        :param tagging_type:str ['pointer_sequence', 'multi_pointer_sequences', 'multi_heads_matrix', 'span_array', 'crf']

        """
        assert sorted(entity_labels) == sorted(list(set(entity_labels)))

        #  entity_labels = ['[unused1]'] + entity_labels
        #  id2label = {i: label for i, label in enumerate(entity_labels)}
        #  label2id = {label: i for i, label in enumerate(entity_labels)}

        from .pointer_sequence import (PointerSequenceTagger,
                                       PointerSequenceExtractor,
                                       PointerSequenceTrainer)
        #  from ..taggers import PointerSequenceTagger  #, MultiPointerSequencesTagger, MultiHeadsMatrixTagger, SpanArrayTagger, CrfTagger
        #  from ..extractors import PointerSequenceExtractor  #, MultiPointerSequencesExtractor, MultiHeadsMatrixExtractor, SpanArrayExtractor, CrfExtractor
        #  from ..trainers import PointerSequenceTrainer  #, MultiPointerSequencesTraiiner, MultiHeadsMatrixTrainer, SpanArrayTrainer, CrfTrainer
        ner_classes = {
            'pointer_sequence': [
                PointerSequenceTagger, PointerSequenceTrainer,
                PointerSequenceExtractor
            ],
            #  'multi_pointer_sequence': [
            #      MultiPointerSequenceTagger, MultiPointerSequenceTrainer,
            #      MultiPointerSequenceCollector
            #  ],
            #  'multi_heads_matrix': [
            #      MultiHeadsMatrixTagger, MultiHeadsMatrixTrainer,
            #      MultiHeadsMatrixCollector
            #  ],
            #  'span_array':
            #  [SpanArrayTagger, SpanArrayTrainer, SpanArrayCollector],
            #  'crf': [CrfTagger, CrfTrainer, CrfCollector]
        }

        if tagging_type in ner_classes:
            tagger_cls, trainer_cls, extractor_cls = ner_classes[tagging_type]
        else:
            raise ValueError(
                f"tagging_type muse be in ['pointer_sequence', 'multi_pointer_sequences', 'multi_heads_matrix', 'span_array', 'crf']"
            )

        trainer = trainer_cls(args, entity_labels)
        tagger = tagger_cls(tokenizer=tokenizer, label2id=args.label2id)
        extractor = extractor_cls()

        super(EntityExtractionModel, self).__init__(args,
                                                    tagger=tagger,
                                                    trainer=trainer,
                                                    extractor=extractor)
        self.entity_labels = entity_labels
        self.tokenizer = tokenizer
        #  self.label2id = label2id

    def train(self, train_dataflow, eval_dataflow):
        args = self.args

        train_features = self.tagger.encode(train_dataflow,
                                            args.train_max_seq_length)
        #  self.tokenizer,
        #  self.label2id
        #  )
        eval_features = self.tagger.encode(eval_dataflow,
                                           args.eval_max_seq_length)
        #  self.tokenizer, self.label2id)
        self.trainer.train(args, train_features, eval_features)

    def evaluate(self, eval_dataflow, model_path=None):
        args = self.args
        model = self.load_model(model_path=args.best_model_path
                                if model_path is None else model_path)
        eval_features = self.tagger.encode(eval_dataflow,
                                           args.train_max_seq_length)
        self.trainer.evaluate(args, model, eval_features)

    def predict(self, test_dataflow, model_path=None):
        args = self.args

        model = self.load_model(model_path=args.best_model_path
                                if model_path is None else model_path)
        test_features = self.tagger.encode(test_dataflow,
                                           args.train_max_seq_length)
        self.trainer.predict(args, model, test_features)

        from ..utils.ner_utils import save_ner_preds
        save_ner_preds(self.args, self.trainer.preds, test_dataflow)
