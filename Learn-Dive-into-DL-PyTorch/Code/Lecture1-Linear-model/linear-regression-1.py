# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : boyuai

"""

线性回归模型从零开始实现

"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import random

print(torch.__version__)

# 生成数据集

# set input feature number
num_inputs = 2
# set example number
num_examples = 1000

# set true weight and bias in order to generate  corresponded label
true_w = [2, -3.4]
true_b = 4.2

features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01,
                                        size=labels.size()), dtype=torch.float32)

# 使用图像来展示生成的数据
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)

# 读取数据集


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)

# 初始化模型参数
w = torch.tensor(np.random.normal(
    0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 定义模型


def linreg(X, w, b):
    return torch.mm(X, w) + b

# 定义损失函数 -- 均方误差函数


def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size()))**2 / 2

# 定义优化函数
# 小批量随机梯度下降


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

# 训练

# super parameters init
batch_size = 10
lr = 0.03
num_epochs = 5

net = linreg

loss = squared_loss

# training
for epoch in range(num_epochs):

    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)
        w.grad.data.zero_()
        b.grad.data.zero_()

    train_l = loss(net(features, w, b), labels)
    print("epoch %d,loss %f" % (epoch + 1, train_l.mean().item()))


w, true_w, true_b
