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
###############################   Uncertainty_based  ###############################
####################################################################################

def Uncertainty(strategy, probs=[], losses=[]):
    '''
    strategy: strategy type，["LeastConfident", "Entropy", "Margin", "MaxLoss"]
    probs: The predicted probability output of the model for examples, for ["LeastConfident", "Entropy", "Margin"]
    return：sample score
    '''
    if probs != []
        probs = np.asarray(probs)
        if strategy == 'LeastConfident':
            score = 1. - probs.max(1)
            score = (score - score.min())/(score.max()-score.min())
        elif strategy == 'Entropy':
            score = -1.*np.sum(probs * np.log(probs + 1e-10), axis=1)
            score = (score - score.min())/(score.max()-score.min())
        elif strategy == 'Margin':
            One_Two = np.argpartition(probs, -2, axis=1)[:,-2:]
            score = []
            for i in range(One_Two.shape[0]):
                score.append(abs(prob[i,One_Two[i,1]] - prob[i,One_Two[i,0]]))
            score = (score - score.min())/(score.max()-score.min())
    elif strategy == 'MaxLoss':
        score = np.asarray(losses)
        score = (score - score.min())/(score.max()-score.min())
    return score


'''
from Gda_tools import *

## 省略

probs = []
idxs = []
for i, (images, labels, indexes) in enumerate(train_loader):
    ind = indexes.cpu().numpy().transpose()
    batch_size = len(ind)

    images = Variable(images).cuda()
    labels = Variable(labels).cuda()

    logits = model(images)

    loss = F.cross_entropy(logits, labels, reduce=False)

    probs.extend(F.softmax(logits, dim=1).cpu().detach().numpy().tolist())
    idxs.extend(indexes.cpu().numpy().tolist())

    loss = torch.mean(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

probs = np.asarray(probs)[np.argsort(idxs)]
scores = Uncertainty("LeastConfident", probs)
'''

def Uncertainty_Img2img(m_output, gt, strategy="SSIM"):
    '''
    m_output: model output
    gt: ground truth
    strategy: strategy type，["SSIM", "PSNR"]
    return：sample score
    '''
    if strategy == "SSIM":
        pixel_scores = ssim(m_output, gt, reduction='none').detach()
    elif strategy == "PSNR":
        pixel_scores = ((gt-m_output)**2).detach()
    scores = pixel_scores.reshape(pixel_scores.shape[0],-1)
    scores = scores.mean(1)
    return scores, pixel_scores

'''
example:

from Gda_tools import *

## 省略

used_scores = [1. for i in range(len(train_dataset))]
for e in range(total_epochs):
    model.train()
    loader_bar = tqdm(train_loader)
        
    next_round_scores = []
    idxs = []

    used_scores = torch.tensor(used_scores).cuda()

    for low, full, target, idx in loader_bar:
        optimizer.zero_grad()
        low = low.to(device)
        full = full.to(device) 
        t = target.to(device)
        res = model(low, full)

        scores, pixel_scores = Uncertainty_Img2img(res, t)

        next_round_scores.extend(scores.cpu().detach().numpy().tolist())
        idxs.extend(idx.cpu().numpy().tolist())
            
        total_loss = (used_scores[idx].float()*mseloss(res, t)).mean()
                
        total_loss.backward()
        optimizer.step()
        
    used_scores = np.asarray(next_round_scores)[np.argsort(idxs)]    
    used_scores = (used_scores-used_scores.min())/(used_scores.max()-used_scores.min())
'''