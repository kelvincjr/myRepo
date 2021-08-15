import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#  class FocalLoss(nn.Module):
#
#      # gamma: focusing parameter
#      # alpha: balance parameter
#      def __init__(self, gamma=2, alpha=0.25):
#          super(FocalLoss, self).__init__()
#
#          self.gamma = gamma
#          self.alpha = alpha
#
#      def forward(self, output, target):
#
#          cross_entropy = F.cross_entropy(output, target)
#          cross_entropy_log = torch.log(cross_entropy)
#          logpt = -F.cross_entropy(output, target)
#          pt = torch.exp(logpt)
#
#          focal_loss = -((1 - pt)**self.gamma) * logpt
#
#          focal_loss = self.alpha * focal_loss
#
#          return focal_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        #  if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        #  print(input)
        #  print(target)
        #  print(f"input.shape: {input.shape}, target.shape: {target.shape}")

        #  if input.dim() > 2:
        #      input = input.view(input.size(0), input.size(1),
        #                         -1)  # N,C,H,W => N,C,H*W
        #      input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
        #      input = input.contiguous().view(
        #          -1, input.size(2))  # N,H*W,C => N*H*W,C

        target = target.view(-1, 1).long()

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


#  class FocalLoss(nn.Module):
#      def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
#          super(FocalLoss, self).__init__()
#          self.gamma = gamma
#          self.alpha = alpha
#          self.reduction = reduction
#
#      def forward(self, output, target):
#          # convert output to pseudo probability
#          out_target = torch.stack([output[i, t] for i, t in enumerate(target)])
#          probs = torch.sigmoid(out_target)
#          focal_weight = torch.pow(1 - probs, self.gamma)
#
#          # add focal weight to cross entropy
#          ce_loss = F.cross_entropy(output,
#                                    target,
#                                    weight=self.alpha,
#                                    reduction='none')
#          focal_loss = focal_weight * ce_loss
#
#          if self.reduction == 'mean':
#              focal_loss = (focal_loss / focal_weight.sum()).sum()
#          elif self.reduction == 'sum':
#              focal_loss = focal_loss.sum()
#
#          return focal_loss

#  class FocalLoss(nn.Module):
#      '''Multi-class Focal loss implementation'''
#      def __init__(self, gamma=2, alpha=None, ignore_index=-100):
#          super(FocalLoss, self).__init__()
#          self.gamma = gamma
#          self.alpha = alpha
#          self.ignore_index = ignore_index
#
#      def forward(self, input, target):
#          """
#          input: [N, C]
#          target: [N, ]
#          """
#          logpt = F.log_softmax(input, dim=1)
#          pt = torch.exp(logpt)
#          logpt = (1 - pt)**self.gamma * logpt
#          loss = F.nll_loss(logpt,
#                            target.long(),
#                            self.alpha,
#                            ignore_index=self.ignore_index)
#          return loss
