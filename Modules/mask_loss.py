import torch
import torch.nn as nn


class MaskedLoss(nn.Module):

    def __init__(self,loss):
        super(MaskedLoss,self).__init__()
        self.loss = loss #nn.MSELoss()

    def forward(self,prediction,target,mask):
            
        # apply the masks
        masked_prediction =  prediction*mask 
        masked_target = target*mask

        return self.loss(masked_prediction,masked_target)
