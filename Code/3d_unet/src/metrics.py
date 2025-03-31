import numpy as np 
import torch 
import sys

def calculate_binary_dice(pred_binary, gt_binary):
    epsilon = 1e-6
    
    intersection = (pred_binary * gt_binary).sum().item()
    sum_cardinality = (pred_binary.sum() + gt_binary.sum()).item()
    return ((2*intersection + epsilon) / (sum_cardinality + epsilon)) 


def calculate_multiclass_dice(pred, gt, num_class): 
    dice_list = []
    
    for c in range(num_class): 
        pred_binary = pred[c, :, :, :]
        gt_binary = gt[c, :, :, :]
        dice = calculate_binary_dice(pred_binary, gt_binary)
        dice_list.append(dice)
        
    return np.mean(dice_list), dice_list