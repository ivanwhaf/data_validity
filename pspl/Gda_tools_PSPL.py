import numpy as np
import torch
from math import exp
import ssl
from random import sample

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, reduction = 'mean'):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if reduction == 'mean':
        return ssim_map.mean()
    elif reduction == 'none':
        return ssim_map
    elif reduction == 'navg':
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        if self.size_average:
            return _ssim(img1, img2, window, self.window_size, channel)
        else:
            return _ssim(img1, img2, window, self.window_size, channel, 'navg')

def ssim(img1, img2, window_size = 11, reduction = 'mean'):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, reduction)

####################################################################################
###############################         PSPL         ###############################
####################################################################################

def PSPL_Img2img(m_output, gt, current_epoch):
    '''
    m_output: model output
    gt: ground truth
    current_epoch: current training epoch
    return：pixel-level sample score
    '''
    alpha = 1
    beta = 0
    maxVal = 2
    sigma = alpha * current_epoch + beta
    gauss = lambda x: torch.exp(-((x+1) / sigma) ** 2) * maxVal
    ssim_val = ssim(gt, m_output, reduction='none').detach()
    pixel_score = gauss(ssim_val).detach()
    return pixel_score

'''
example:


from Gda_tools import *

## 省略

for e in range(total_epochs):
    model.train()
    loader_bar = tqdm(train_loader)
        
    for low, full, target in loader_bar:
        optimizer.zero_grad()
        low = low.to(device)
        full = full.to(device)  
        t = target.to(device) 
        res = model(low, full)

        pixel_scores = PSPL_Img2img(res, t, e+1)

        total_loss = mseloss(res*pixel_scores, t*pixel_scores).mean()
                
        total_loss.backward()
        optimizer.step()
'''


