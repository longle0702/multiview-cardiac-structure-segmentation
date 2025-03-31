import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss3D(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceLoss3D, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, gt):
        intersection = (pred * gt).sum(dim = (1,2,3,4))

        pred_sum = (pred * pred).sum(dim = (1,2,3,4))
        gt_sum = (gt * gt).sum(dim = (1,2,3,4))

        numerator = 2 * intersection + self.epsilon
        denom = pred_sum + gt_sum + self.epsilon
        
        dice = numerator/denom
        dice_loss = 1 - dice.mean()
  
        return dice_loss