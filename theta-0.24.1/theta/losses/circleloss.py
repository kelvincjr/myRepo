#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------
#  from typing import Tuple
#  import torch
#  import torch.nn as nn
#  from torch import nn, Tensor
#  import torch.nn.functional as F
#
#  class CircleLoss(nn.Module):
#      def __init__(self, gamma=32, m=0.25, similarity='cos', **kwargs):
#          super(CircleLoss, self).__init__()
#          self.gamma = gamma
#          self.m = m
#          self.similarity = similarity
#
#      def forward(self, feats, labels):
#          assert feats.size(0) == labels.size(0), \
#              f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
#
#          m = labels.size(0)
#          mask = labels.expand(m, m).t().eq(labels.expand(m, m)).float()
#          pos_mask = mask.triu(diagonal=1)
#          neg_mask = (mask - 1).abs_().triu(diagonal=1)
#          if self.similarity == 'dot':
#              sim_mat = torch.matmul(feats, torch.t(feats))
#          elif self.similarity == 'cos':
#              feats = F.normalize(feats)
#              sim_mat = feats.mm(feats.t())
#          else:
#              raise ValueError('This similarity is not implemented.')
#
#          pos_pair_ = sim_mat[pos_mask == 1]
#          neg_pair_ = sim_mat[neg_mask == 1]
#
#          alpha_p = torch.relu(-pos_pair_ + 1 + self.m)
#          alpha_n = torch.relu(neg_pair_ + self.m)
#          margin_p = 1 - self.m
#          margin_n = self.m
#          loss_p = torch.sum(
#              torch.exp(-self.gamma * alpha_p * (pos_pair_ - margin_p)))
#          loss_n = torch.sum(
#              torch.exp(self.gamma * alpha_n * (neg_pair_ - margin_n)))
#          loss = torch.log(1 + loss_p * loss_n)
#          return loss

# -------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


def feature_similarity(x, label, similarity='cos'):
    n = label.size(0)
    mask = label.expand(n, n).t().eq(label.expand(n, n)).float()
    pos_mask = mask.triu(diagonal=1)
    neg_mask = (mask - 1).abs_().triu(diagonal=1)

    # inspired by attention
    x = x.squeeze()
    if similarity == 'dot':
        sim_mat = x.matmul(x.transpose(0, 1))
    elif similarity == 'cos':
        x = F.normalize(x)
        sim_mat = x.matmul(x.transpose(0, 1))

    return sim_mat[pos_mask == 1], sim_mat[neg_mask == 1]


class CircleLoss(nn.Module):
    def __init__(self, gamma=80, m=0.25, similarity='cos'):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.similarity = similarity
        self.soft_plus = nn.Softplus()

    def forward(self, x, label):
        sp, sn = feature_similarity(x, label, similarity=self.similarity)
        ap = torch.relu(-sp.detach() + 1 + self.m)
        an = torch.relu(sn.detach() + self.m)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss_p = torch.sum(torch.exp(logit_p))
        loss_n = torch.sum(torch.exp(logit_n))

        loss = torch.log(1 + loss_p * loss_n)

        return loss


# -------------------------------------------------------
#  from typing import Tuple
#
#  import torch
#  from torch import nn, Tensor
#
#
#  def convert_label_to_similarity(normed_feature: Tensor,
#                                  label: Tensor) -> Tuple[Tensor, Tensor]:
#      similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
#      label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
#
#      positive_matrix = label_matrix.triu(diagonal=1)
#      negative_matrix = label_matrix.logical_not().triu(diagonal=1)
#
#      similarity_matrix = similarity_matrix.view(-1)
#      positive_matrix = positive_matrix.view(-1)
#      negative_matrix = negative_matrix.view(-1)
#      return similarity_matrix[positive_matrix], similarity_matrix[
#          negative_matrix]
#
#
#  class CircleLoss(nn.Module):
#      #  def __init__(self, m: float, gamma: float) -> None:
#      def __init__(self, gamma: float = 32, m: float = 0.25) -> None:
#          super(CircleLoss, self).__init__()
#          self.m = m
#          self.gamma = gamma
#          self.soft_plus = nn.Softplus()
#
#      def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
#          ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
#          an = torch.clamp_min(sn.detach() + self.m, min=0.)
#
#          delta_p = 1 - self.m
#          delta_n = self.m
#
#          logit_p = -ap * (sp - delta_p) * self.gamma
#          logit_n = an * (sn - delta_n) * self.gamma
#
#          loss = self.soft_plus(
#              torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
#
#          return loss
#
#
#  if __name__ == "__main__":
#      feat = nn.functional.normalize(torch.rand(256, 64, requires_grad=True))
#      lbl = torch.randint(high=10, size=(256, ))
#
#      inp_sp, inp_sn = convert_label_to_similarity(feat, lbl)
#
#      criterion = CircleLoss(m=0.25, gamma=256)
#      circle_loss = criterion(inp_sp, inp_sn)
#
#      print(circle_loss)
#
#  #  def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
#  #      similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
#  #      label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
#  #
#  #      positive_matrix = label_matrix.triu(diagonal=1)
#  #      negative_matrix = label_matrix.logical_not().triu(diagonal=1)
#  #
#  #      similarity_matrix = similarity_matrix.view(-1)
#  #      positive_matrix = positive_matrix.view(-1)
#  #      negative_matrix = negative_matrix.view(-1)
#  #      return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]
#  #
#  #
#  #  class CircleLoss(nn.Module):
#  #      def __init__(self, gamma: float, m: float) -> None:
#  #          super(CircleLoss, self).__init__()
#  #          self.m = m
#  #          self.gamma = gamma
#  #          self.soft_plus = nn.Softplus()
#  #
#  #      #def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
#  #  def forward(self, feats, labels):
#  #      sp, sn = convert_label_to_similarity(feats, labels)
#  #          sp = nn.functional.normalize(sp)
#  #          sp, sn = convert_label_to_similarity(sp, sn)
#  #          ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
#  #          an = torch.clamp_min(sn.detach() + self.m, min=0.)
#  #
#  #          delta_p = 1 - self.m
#  #          delta_n = self.m
#  #
#  #          logit_p = - ap * (sp - delta_p) * self.gamma
#  #          logit_n = an * (sn - delta_n) * self.gamma
#  #
#  #          loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
#  #
#  #          return loss
