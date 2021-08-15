#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


def common_batch_encode(texts, label2id, tokenizer, max_seq_length):
    all_encodes = tokenizer.batch_encode(texts, add_special_tokens=True)
    all_tokens = all_encodes['tokens']
    all_token_offsets = all_encodes['offsets']
    all_token2char = all_encodes['token2char']
    all_char2token = all_encodes['char2token']

    all_input_ids = all_encodes['ids']
    all_attention_mask = all_encodes['attention_mask']
    all_token_type_ids = all_encodes['type_ids']

    all_input_lens = [len(tokens) for tokens in all_tokens]

    all_padding_lens = [max_seq_length - n for n in all_input_lens]
    for i, (input_ids, attention_mask, token_type_ids, token2char,
            token_offsets, padding_length) in enumerate(
                tqdm(zip(all_input_ids, all_attention_mask, all_token_type_ids,
                         all_token2char, all_token_offsets, all_padding_lens),
                     desc="common_batch_encode")):

        if padding_length < 0:
            all_tokens[i] = all_tokens[i][:max_seq_length - 1] + ['[SEP]']
            all_input_ids[i] = input_ids[:max_seq_length - 1] + [102]
            all_attention_mask[i] = attention_mask[:max_seq_length]
            all_token_type_ids[i] = token_type_ids[:max_seq_length]
            all_token2char[i] = token2char[:max_seq_length]
            all_token_offsets[i] = token_offsets[:max_seq_length - 1] + [(0, 0)
                                                                         ]
            #  logger.warning(f"all_tokens[i]: {all_tokens[i]}")
            #  logger.debug(f"all_input_ids[i]: {all_input_ids[i]}")
            #  logger.debug(f"all_token_offsets[i]: {all_token_offsets[i]}")

        else:
            all_input_ids[i] = input_ids + [0] * padding_length
            all_attention_mask[i] = attention_mask + [0] * padding_length
            all_token_type_ids[i] = token_type_ids + [0] * padding_length
            all_token2char[i] = token2char + [0] * padding_length
            all_token_offsets[i] = token_offsets + [(0, 0)] * padding_length
        all_input_lens[i] = len(input_ids)

    #  logger.warning(f"all_input_ids: {all_input_ids}")
    all_input_ids = torch.from_numpy(np.array(all_input_ids, dtype=np.int64))
    all_attention_mask = torch.from_numpy(
        np.array(all_attention_mask, dtype=np.int64))
    all_token_type_ids = torch.from_numpy(
        np.array(all_token_type_ids, dtype=np.int64))
    all_input_lens = torch.from_numpy(np.array(all_input_lens, dtype=np.int64))
    all_token_offsets = torch.from_numpy(
        np.array(all_token_offsets, dtype=np.int64))

    return (all_tokens, all_token2char, all_char2token, all_input_ids,
            all_attention_mask, all_token_type_ids, all_input_lens,
            all_token_offsets)


def common_to_tensors(all_input_ids, all_attention_mask, all_token_type_ids,
                      all_input_lens, all_token_offsets):
    all_input_ids = torch.from_numpy(np.array(all_input_ids, dtype=np.int64))
    all_attention_mask = torch.from_numpy(
        np.array(all_attention_mask, dtype=np.int64))
    all_token_type_ids = torch.from_numpy(
        np.array(all_token_type_ids, dtype=np.int64))
    all_input_lens = torch.from_numpy(np.array(all_input_lens, dtype=np.int64))
    all_token_offsets = torch.from_numpy(
        np.array(all_token_offsets, dtype=np.int64))

    return all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets


def encode_examples(examples, label2id, tokenizer, max_seq_length):

    num_labels = len(label2id)
    texts = [e.text_a[:max_seq_length - 2] for e in examples]

    all_tokens, all_token2char, all_char2token, all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets = common_batch_encode(
        texts, label2id, tokenizer, max_seq_length)

    #  all_labels = [label2id[e.label] if e.label else 0 for e in examples]
    all_labels = []
    for e in examples:
        if isinstance(e.label, list):
            targets = [0] * len(label2id)
            for x in e.label:
                targets[label2id[x]] = 1
            all_labels.append(targets)
        else:
            if e.label:
                all_labels.append(label2id[e.label])
            else:
                all_labels.append(0)

    all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens, all_token_offsets = common_to_tensors(
        all_input_ids, all_attention_mask, all_token_type_ids, all_input_lens,
        all_token_offsets)

    all_labels = torch.from_numpy(np.array(all_labels, dtype=np.int64))

    all_features = [
        InputFeature(input_ids=input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids,
                     input_len=input_len,
                     token_offsets=token_offsets,
                     label=label,
                     text=text) for input_ids, attention_mask,
        token_type_ids, input_len, token_offsets, label, text in zip(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_input_lens, all_token_offsets, all_labels, texts)
    ]

    return all_features
