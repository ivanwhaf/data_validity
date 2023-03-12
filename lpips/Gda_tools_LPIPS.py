import numpy as np
import torch
from math import exp
import lpips
import ssl
from random import sample


####################################################################################
###############################         LPIPS        ###############################
####################################################################################

ssl._create_default_https_context = ssl._create_unverified_context
def LPIPS_Img2img(m_output, gt):
    '''
    m_output: model output
    gt: ground truth
    return：sample score
    '''
    alpha = 0.1
    fn_vgg = lpips.LPIPS(net='vgg', verbose=False).cuda()
    score = alpha*fn_vgg(m_output, gt)
    return score

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

        scores = LPIPS_Img2img(res, t)

        total_loss = (mseloss(res, t) + scores).mean()
                
        total_loss.backward()
        optimizer.step()
'''
