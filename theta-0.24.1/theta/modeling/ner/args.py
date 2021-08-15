#!/usr/bin/env python
# -*- coding: utf-8 -*-

from loguru import logger
from ..common_args import get_main_args


def add_modeling_args(parser):
    parser.add_argument('--markup',
                        default='bios',
                        type=str,
                        choices=['bios', 'bio'])
    parser.add_argument('--ner_type',
                        default='crf',
                        type=str,
                        choices=['crf', 'span'])
    parser.add_argument("--autofix",
                        action="store_true",
                        help="Auto fix CRF label errors.")
    parser.add_argument(
        "--bios_file",
        type=str,
        default=None,
        help="The bios file path.",
    )
    parser.add_argument(
        "--crf_type",
        type=str,
        default='ncrfpp',
        help=
        "CRF type. ['ncrfpp', 'pytorch-crf', 'old_crf', 'new_crf', 'lstm_crf']"
    )
    parser.add_argument("--no_crf_loss",
                        action="store_true",
                        help="Do not calculate CRF loss.")

    return parser


def get_args(experiment_params = None, special_args: list = None):
    return get_main_args(add_modeling_args, 
                         experiment_params=experiment_params,
                         special_args = special_args)
