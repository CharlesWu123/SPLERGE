# -*- coding: utf-8 -*-
# cython: language_level=3
'''
@version : 0.1
@Author : Charles
@Time : 2020/3/12 下午1:59
@File : params.py
'''
import os

# 图像预处理
std = 0.193
mean = 0.588
maxW = 600
maxH = 600

# 训练
batchSize = 1
workers = 1
lr = 0.00075    # 学习率
beta1 = 0.9
niter = 300     # 迭代次数
saveModel = 10  # 每多少步保存一次模型
displayInterval = 20    # 打印间隔
trained_model = ''      # 继续训练的模型
experiment = os.path.abspath('./trained_models')    # 模型保存的文件夹路径

os.makedirs(experiment, exist_ok=True)