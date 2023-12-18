import torch.nn as nn
from segmentation_models_pytorch.utils.losses import DiceLoss

class Dice_BCE_Loss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(Dice_BCE_Loss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, input, target):
        return self.bce_weight * self.dice_loss(input, target) + self.dice_weight * self.bce_loss(input, target)
