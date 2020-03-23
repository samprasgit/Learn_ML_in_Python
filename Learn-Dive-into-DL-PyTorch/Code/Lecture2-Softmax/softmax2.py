# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : boyuai
# @Editor  : sublime 3
#
"""
softamx pytorch简洁实现
"""

import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
path = '../../dataset'

print(torch.__version__)

# 初始化参数和获取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root=path)

# 定义网络模型
num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):  # x 的形状: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y

# net = LinearNet(num_inputs, num_outputs)


class FlattenLayer(nn.Module):

    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x 的形状: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

from collections import OrderedDict
net = nn.Sequential(
    # FlattenLayer(),
    # LinearNet(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))])  # 或者写成我们自己定义的 LinearNet(num_inputs, num_outputs) 也可以
)

# 初始化模型参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 定义损失函数
loss = nn.CrossEntropyLoss()
# 下面是他的函数原型
# class torch.nn.CrossEntropyLoss(weight=None, size_average=None,
# ignore_index=-100, reduce=None, reduction='mean')


# 定义优化函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
# 下面是函数原型
# class torch.optim.SGD(params, lr=, momentum=0, dampening=0,
# weight_decay=0, nesterov=False)

print("开始训练")


def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的
    # 因为一般用PyTorch计算loss时就默认是沿batch维求平均,而不是sum。
    # 这个无大碍，根据实际情况即可
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

num_epochs = 5
train_ch3(net, train_iter, test_iter, loss, num_epochs,
          batch_size, None, None, optimizer)
