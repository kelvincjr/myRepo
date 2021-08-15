#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


class Swish(nn.Module):

    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, x):
        '''
        Forward pass of the function.
        '''
        return swish(x)
