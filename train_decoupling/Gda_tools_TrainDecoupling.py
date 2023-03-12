import numpy as np
import torch
from math import exp
import ssl
from random import sample
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler


####################################################################################
###############################      Train Decoupling     ##########################
####################################################################################


# 常规训练流程
def train(epochs, model, loader):
    pass

# 该函数用于丢样本，创建一个 Sampler，分数低于某阈值时采样权重调整为 0，即不加入训练
def clean_dataset(dataset_train, score, threshold):
    weights = [1 if score>= threshold else 0 for score_idx in score]
    sampler_train = WeightedRandomSampler(weights, num_samples=len(dataset_train), replacement=True)
    return  DataLoader(dataset_train, batch_size=16, drop_last=True, sampler=sampler_train)

# 先训全量数据，再训丢弃完的数据
# train_ratio 代表全量数据的训练轮数占总训练轮数的比例

def trainDecoupling(model, dataset_train, train_ratio, overall_epochs, clean_score, threshold):
    overall_dataloader = DataLoader(dataset_train, batch_size=16, drop_last=True)
    train(
        epochs=int(overall_epochs * train_ratio), 
        model=model,
        loader=overall_dataloader
    )

    cleaned_dataloader = clean_dataset(dataset_train, clean_score, threshold)
    
    train(
        epochs=overall_epochs - int(overall_epochs * train_ratio), 
        model=model,
        loader=cleaned_dataloader
    )

'''
from Gda_tools import *

model =                 # 加载所训练的模型
train_dataset =         # 加载训练使用的 dataset
clean_score =           # 基于打分方法生成的分数，归一化至 [0,1] 之间
overall_epochs = 100    # 总共训练轮数为 100 轮
train_ratio = 0.5       # 全量数据训练比例为 train_ratio, 清洗后数据的训练比例为(1 - train_ratio)
threshold = 0.2         # 分数低于该阈值的样本被丢弃

trainDecoupling(model, dataset, train_ratio, overall_epochs, clean_score, threshold)
'''

