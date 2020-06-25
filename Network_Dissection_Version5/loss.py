import torch.nn as nn
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from IPython import embed

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true, threshold=0.5):
        assert y_pred.size() == y_true.size()
        chan_count = y_pred.shape[1]
        y_pred = y_pred.permute(1,0,2,3).contiguous().view(chan_count,-1)
        y_true = y_true.permute(1,0,2,3).contiguous().view(chan_count,-1)
        intersection = (y_pred * y_true).sum(dim=1)
        dsc = ( 2. * intersection + self.smooth ) / ( (y_pred + y_true).sum(dim=1) + self.smooth )
        dsc = dsc.sum()/chan_count
    
        return 1. - dsc


# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1.0):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
    
#     def forward(self, y_pred, y_true):
#         assert y_pred.size() == y_true.size()
#         chan_count = y_pred.shape[1]
#         y_pred = y_pred.permute(1,0,2,3).contiguous().view(chan_count,-1)
#         y_true = y_true.permute(1,0,2,3).contiguous().view(chan_count,-1)
#         pixel_count = y_pred.shape[1]
    
#         CE_loss = -(y_true*torch.log(y_pred))
    
#         CE_loss = CE_loss.sum(0)
    
#         return CE_loss.mean()