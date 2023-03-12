import numpy as np
import torch
from math import exp
import ssl
from random import sample


####################################################################################
###############################          SPE         ###############################
####################################################################################

def SPE(losses, idxs, current_epoch, total_epochs, k=5):
    '''
    losses: The loss value of a certain round for all samples
    idxs: sample index
    current_epoch: current training epoch
    total_epochs: total training epochs
    k: the number of bins
    return：sample score
    '''
    ## split bins
    sort_idx = np.argsort(losses)
    min_loss = losses[sort_idx[0]]
    max_loss = losses[sort_idx[-1]]
    bins_loss = [[] for i in range(k)]
    bins_idx = [[] for i in range(k)]
    split_point = [0 for i in range(k)]
    step = (max_loss-min_loss)/k
    count = 0
    while losses[sort_idx[0]] > (count + 1)*step + min_loss:
        count = count + 1
    for i in sort_idx:
        if losses[i] > (count + 1)*step + min_loss and count + 1 < k:
            split_point[count] = idxs[i]
            count = count + 1
        bins_loss[count].append(losses[i])
        bins_idx[count].append(idxs[i])
    ## resample
    bins_p = []
    alpha = np.tan((current_epoch*np.pi)/(2*total_epochs))
    for i in range(k):
        mean = np.mean(bins_loss[i])
        bins_p.append(1/(mean+alpha))
    p_sum = np.sum(bins_p)+1e-5
    num = int(len(idxs)/4)
    select_idx = []
    for i in range(k):
        bins_p[i] = int((bins_p[i]/p_sum)*num)
        select_idx.extend(np.random.choice(bins_idx[i], bins_p[i]))
    scores = []
    for idx in idxs:
        if idx in select_idx:
            scores.append(1.)
        else:
            scores.append(0.)
    return scores

'''
from Gda_tools import *

## 省略

losses = []
idxs = []
for i, (images, labels, indexes) in enumerate(train_loader):
    ind = indexes.cpu().numpy().transpose()
    batch_size = len(ind)

    images = Variable(images).cuda()
    labels = Variable(labels).cuda()

    logits = model(images)

    loss = F.cross_entropy(logits, labels, reduce=False)

    losses.extend(loss.cpu().detach().numpy().tolist())
    idxs.extend(indexes.cpu().numpy().tolist())

    loss = torch.mean(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

losses = np.asarray(losses)[np.argsort(idxs)]
scores = SPE(losses, current_epoch, total_epochs)
'''
