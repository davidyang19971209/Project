import torch.nn as nn
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from IPython import embed

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0


    def forward(self, y_true, y_pred, epsilon=1e-6): 
        ''' 
        Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
        Assumes the `channels_last` format.
    
        # Arguments
            y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
            y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
            epsilon: Used for numerical stability to avoid divide by zero errors
        
        # References
            V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
            https://arxiv.org/abs/1606.04797
            More details on Dice loss formulation 
            https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
            
            Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
        '''    
        # skip the batch and class axis for calculating Dice score
    
        axes = tuple(range(2, len(y_pred.shape))) 
        numerator = 2*(y_pred * y_true).sum(axes)
        denominator = ((y_pred) + (y_true)).sum(axes)
        dsc = (numerator / (denominator + epsilon))
        
        # return 1- dsc.mean(0)[1:].mean()
        return 1 - dsc.mean() # average over classes and batch
        
        
        # numerator = 2*(y_pred * y_true).sum()
        # denominator = ((y_pred) + (y_true)).sum()
        # dsc = (numerator / (denominator + epsilon))
        # return 1 - dsc # average over classes and batch
 
        # Generalized dice
        # y_true = y_true.permute(0,2,3,1)
        # y_pred = y_pred.permute(0,2,3,1)
        
        # num_class = y_true.shape[-1]
        
        # y_true = y_true.reshape((-1,num_class))
        # y_pred = y_pred.reshape((-1,num_class))
        
        # sum_t = y_true.sum(-2)
        # sum_p = y_pred.sum(-2)
        # sum_tp = (y_true*y_pred).sum(-2)
        
        # weights = (sum_t.pow(2)+1e-7).pow(-1)
        # generalized_dice = (2*(weights*sum_tp).sum())/((weights*(sum_t+sum_p)).sum())
        
        # return 1-generalized_dice
        
        # Generalized dice_V2
        
        
        # axes = tuple(range(2, len(y_pred.shape))) 
        # sum_t = y_true.sum(axes)
        # sum_p = y_pred.sum(axes)
        # sum_tp = (y_true*y_pred).sum(axes)
        # weights = (sum_t.pow(2)+1e-7).pow(-1)
        # generalized_dice = (2*(weights*sum_tp).sum())/((weights*(sum_t+sum_p)).sum())

        # return 1-generalized_dice