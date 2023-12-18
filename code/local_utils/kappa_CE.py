#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:53:41 2023
@author: Boxiang Yun   School:ECNU   Email:boxiangyun@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Kappa_CE(nn.Module):
    def __init__(self, reduction='mean'):
        super(Kappa_CE, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        n_classes = input.size(1)
        onehot = F.one_hot(target,n_classes)
        logp = F.log_softmax(input, dim=1)
        weight = (torch.abs((torch.argmax(input,dim=1) - target)) + 1).view(-1,1)
        loss = torch.sum(-logp * onehot * weight, dim=1)
        
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(
                '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')
            
if __name__ == '__main__':
    kloss = Kappa_CE()
    a = torch.tensor([5,5,5,5])
    b = torch.tensor([[0.1,0,0,0,0,0.9],
                      [0,0,0,0,0.9,0.1],
                      [0,0,0,0.9,0,0.1],
                      [0,0,0.9,0,0,0.1]])
    print(kloss(b,a))