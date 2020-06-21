import torch.nn as nn
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from IPython import embed

# class DiceLoss(nn.Module):

#     def __init__(self):
#         super(DiceLoss, self).__init__()
#         self.smooth = 1.0


#     def forward(self, y_true, y_pred, epsilon=1e-6): 
#         ''' 
#         Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
#         Assumes the `channels_last` format.
    
#         # Arguments
#             y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
#             y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
#             epsilon: Used for numerical stability to avoid divide by zero errors
        
#         # References
#             V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
#             https://arxiv.org/abs/1606.04797
#             More details on Dice loss formulation 
#             https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
            
#             Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
#         '''    
#         # skip the batch and class axis for calculating Dice score
    
#         axes = tuple(range(2, len(y_pred.shape))) 
#         numerator = 2*(y_pred * y_true).sum(axes)
#         denominator = ((y_pred) + (y_true)).sum(axes)
#         dsc = (numerator / (denominator + epsilon))
        
#         # return 1- dsc.mean(0)[1:].mean()
#         return 1 - dsc.mean() # average over classes and batch
        
        
#         # numerator = 2*(y_pred * y_true).sum()
#         # denominator = ((y_pred) + (y_true)).sum()
#         # dsc = (numerator / (denominator + epsilon))
#         # return 1 - dsc # average over classes and batch
 

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1.0):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, y_pred, y_true, threshold=0.5):
#         assert y_pred.size() == y_true.size()
#         chan_count = y_pred.shape[1]
#         y_pred = y_pred.permute(1,0,2,3).contiguous().view(chan_count,-1)
#         y_true = y_true.permute(1,0,2,3).contiguous().view(chan_count,-1)
#         intersection = (y_pred * y_true).sum(dim=1)
#         dsc = ( 2. * intersection + self.smooth ) / ( (y_pred + y_true).sum(dim=1) + self.smooth )
#         dsc = dsc.sum()/chan_count
    
#         return 1. - dsc


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        chan_count = y_pred.shape[1]
        y_pred = y_pred.permute(1,0,2,3).contiguous().view(chan_count,-1)
        y_true = y_true.permute(1,0,2,3).contiguous().view(chan_count,-1)
        pixel_count = y_pred.shape[1]
    
        CE_loss = -(y_true*torch.log(y_pred))
    
        CE_loss = CE_loss.sum(0)
    
        return CE_loss.mean()