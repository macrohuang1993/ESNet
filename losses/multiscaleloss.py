import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F


def EPE(input_flow, target_flow, size_average = True):
    target_valid = (target_flow < 192) & (target_flow > 0) 
    return F.l1_loss(input_flow[target_valid], target_flow[target_valid], size_average=size_average)
def smoothL1(input_flow, target_flow, size_average = True):
    """
    When size_average = True (used in FADNet), the loss is averaged over pixels.
    When size_average = False (for PWCNet), the loss is summed over pixels. 
    """
    target_valid = (target_flow < 192) & (target_flow > 0) 
    return F.smooth_l1_loss(input_flow[target_valid], target_flow[target_valid], size_average=size_average)
def robust_EPE(input_flow, target_flow, _div_flow = 0.05):
    N = input_flow.shape[0]
    target_flow = target_flow * _div_flow
    target_valid = (target_flow < 192 * _div_flow) & (target_flow > 0) 
    return torch.pow((torch.abs(target_flow[target_valid] - input_flow[target_valid]) + 0.01),0.4).sum()/N 

class MultiScaleLoss(nn.Module):

    def __init__(self, scales, downscale, weights=None, train_loss = 'smoothL1', test_loss='L1', mask=False):
        """
        downscale is 1 in fadnet and has no effect. 
        test loss specifies what loss to use in the test case. L1 is used in FADNet. During training, the loss is calculated 
        using EPE defined above( smoothL1 loss).
        """
        super(MultiScaleLoss, self).__init__()
        self.downscale = downscale
        self.mask = mask
        self.weights = torch.Tensor(scales).fill_(1).cuda() if weights is None else torch.Tensor(weights).cuda()
        assert(len(self.weights) == scales)
        if train_loss == 'smoothL1':
            self.train_loss = smoothL1
        elif train_loss == 'L1':
            self.train_loss = EPE
        else:
            raise NotImplementedError
        if type(test_loss) is str:

            if test_loss == 'L1':
                self.test_loss = nn.L1Loss()
            else:
                raise NotImplementedError
        else:
            self.test_loss = test_loss
        self.multiScales = [nn.AvgPool2d(self.downscale*(2**i), self.downscale*(2**i)) for i in range(scales)]

        print('self.multiScales: ', self.multiScales, ' self.downscale: ', self.downscale)

    def forward(self, input, target):
        #print(len(input))
        if (type(input) is tuple) or (type(input) is list):
            out = 0
            for i, input_ in enumerate(input):

                target_ = self.multiScales[i](target)

                if self.mask:
                    # work for sparse
                    mask = target > 0
                    mask.detach_()
                    
                    mask = mask.type(torch.cuda.FloatTensor)
                    pooling_mask = self.multiScales[i](mask) 

                    # use unbalanced avg
                    target_ = target_ / pooling_mask # div by 0 generates nan


                    mask = target_ > 0 # exclude nan pixel
                    mask.detach_()
                    input_ = input_[mask]
                    target_ = target_[mask]

                loss_ = self.train_loss(input_, target_)
                out += self.weights[i] * loss_
        else:
            #This is used in trainer validate for calculating val loss, but val loss is not recored or used anywhere.
            out = self.test_loss(input, self.multiScales[0](target))
        return out

def multiscaleloss(scales=5, downscale=4, weights=None, train_loss = 'smoothL1', test_loss='L1', mask=False):
    if weights is None:
        weights = (0.005, 0.01, 0.02, 0.08, 0.32)
    if scales == 1 and type(weights) is not tuple:
        weights = (weights, )
    return MultiScaleLoss(scales, downscale, weights, train_loss, test_loss, mask)
