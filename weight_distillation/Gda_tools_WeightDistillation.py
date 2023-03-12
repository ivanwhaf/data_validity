import numpy as np
import torch
from math import exp
import ssl
from random import sample

####################################################################################
###############################      WeightDistillation     ########################
####################################################################################

class WeightDistillation():
    def __init__(self, overall_iteraion):
        # self.T 是超参数，用于教师和学生融合，训练的初始轮数依赖教师权重，随着训练进行，学生权重的比例逐渐上升
        self.T = list(np.linspace(0, 0, int(overall_iteraion * 0.1)))
        self.T.extend(list(np.linspace(0.05, 0.5, int(overall_iteraion * 0.9))))
    
    
    def get_score(self, logit):
        # 依照自己的方法通过 logit 获得分数
        pass

    def get_T(self, curr_iteration):
        return self.T[curr_iteration]

    def weight_kd(self, teacher_weight, student_logit, curr_iteration):
        # 分别获得教师权重和学生权重
        teacher_score = teacher_weight
        student_score = self.get_score(student_logit)

        # 利用温度系数进行权重融合
        T = self.get_T(curr_iteration)
        sample_weight = T * student_score + (1 - T) * teacher_score

        return sample_weight


'''
from Gda_tools import *
model =                 # 加载模型
train_dataloader =      # 加载训练 dataloader
optimizer =             # 定义优化器
clean_threshold = 0.2   # 清洗阈值
def get_teacher_weights():
    pass

def clean_dataset(dataset, score, threshold):
    # 教师分数大于一定阈值则保留
    return dataset[score > threshold]

def train():
    teacher_weights = get_teacher_weights()

    # 1、利用权重清洗数据集
    dataset = clean_dataset(dataset, teacher_weight, clean_threshold)

    # 2、利用权重加权学习

    # 实例化权重蒸馏类
    WeightDistillationTool = WeightDistillation()

    for curr_iteration, (images, labels, indexes) in enumerate(train_dataloader):
        indexes = indexes.cpu().numpy().transpose()
        teacher_weight = teacher_weights[indexes]
        
        images = images.cuda()
        labels = labels.cuda()
        student_logits = model(images)

        loss = F.cross_entropy(student_logits, labels, reduce=False)

        # 传入教师和学生的预测logits，获得样本权重
        sample_weight = WeightDistillationTool.weight_kd(teacher_weight, student_logits, curr_iteration)

        loss = loss * sample_weight
        loss = torch.mean(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
'''
