#  import torch
#  from torch import nn
#
#  class DiceLoss(nn.Module):
#      """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
#      Useful in dealing with unbalanced data
#      """
#      def __init__(self):
#          super(DiceLoss, self).__init__()
#
#      def forward(self,input, target):
#          '''
#          input: [N, C]
#          target: [N, ]
#          '''
#          prob = torch.softmax(input, dim=1)
#          prob = torch.gather(prob, dim=1, index=target.unsqueeze(1))
#          dsc_i = 1 - ((1 - prob) * prob) / ((1 - prob) * prob + 1)
#          dice_loss = dsc_i.mean()
#          return dice_loss

import torch
import torch.nn as nn
import numpy as np


def make_one_hot(vol, mask):
    '''
    :param vol: the segmentation map,[N, 1, :, :, :],N: batch size
    :param mask: can be a python list including all kinds of pixel values of your segmentation map; e.g. mask=[1,2,3]
    :return: [N, len(mask), :, :, :]
    '''

    lens = len(mask)
    shape = np.array(vol.shape)
    shape[1] = lens
    shape = tuple(shape)
    result = torch.zeros(shape)

    for idx, label in enumerate(mask):
        tmp = vol == label
        result[:, idx] = tmp

    return result


class DiceLoss(nn.Module):
    '''
    vol1,vol2: need to make one hot first
    '''
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, vol1, vol2):
        shape = vol1.shape
        total_loss = 0
        for i in range(shape[1]):
            top = 2 * torch.sum(torch.mul(vol1[:, i], vol2[:, i]), dtype=float)

            bottom = torch.sum(vol1[:, i], dtype=float) + torch.sum(
                vol2[:, i], dtype=float)
            bottom = torch.max(bottom, (torch.ones_like(bottom, dtype=float) *
                                        self.epsilon))  # add epsilon.
            loss_tmp = -1 * (top / bottom)
            total_loss += loss_tmp

        return total_loss / shape[1]


def dice_loss_binary(vol1, vol2):
    top = 2 * torch.sum(torch.mul(vol1, vol2), dtype=float)
    bottom = torch.sum(vol1, dtype=float) + torch.sum(vol2, dtype=float)
    bottom = torch.max(
        bottom, (torch.ones_like(bottom, dtype=float) * 1e-5))  # add epsilon.
    loss = -1 * (top / bottom)

    return loss
