import numpy as np
import torch
from math import exp
import ssl
from random import sample


####################################################################################
###############################      Attack_based    ###############################
####################################################################################

def Attack_based(data_in, data_out, adv_model, adv_loss, epsilon=0.01, flag=False):
    '''
    data_in: data input
    data_out: data output
    adv_model: adversarial model
    adv_loss: adversarial loss
    epsilon: adversarial attack strength
    flag: flag is True means the data_in is [data1, data2]
    return：sample score
    '''
    if flag:
        x1 = data[1]
        x2 = data[0]
        x1.requires_grad = True
        adv_res = adv_model(x2, x1)
    else:
        x1 = data
        x1.requires_grad = True
        adv_res = adv_model(x1)
    adv_model.zero_grad()
    loss = adv_loss(adv_res, y)
    loss.backward()
    pixel_score = x1.grad.sign()

    score = pixel_score.reshape(pixel_score.shape[0], -1).mean(1)
    return score

'''
from Gda_tools import *

## 省略

adv_model = # 加载对抗模型
adv_loss =  # 定义对抗损失

for e in range(total_epochs):
    model.train()
    loader_bar = tqdm(train_loader)
        
    used_scores = []
    idxs = []

    for low, full, target, idx in loader_bar:
        optimizer.zero_grad()
        low = low.to(device)
        full = full.to(device) 
        t = target.to(device)
        res = model(low, full)

        scores = Attack_based(low, full, t, adv_model, adv_loss)

        used_scores.extend(scores.cpu().detach().numpy().tolist())
        idxs.extend(idx.cpu().numpy().tolist())
            
        total_loss = mseloss(res, t).mean()
                
        total_loss.backward()
        optimizer.step()
        
    used_scores = np.asarray(used_scores)[np.argsort(idxs)]    
    used_scores = (used_scores-used_scores.min())/(used_scores.max()-used_scores.min())
'''
