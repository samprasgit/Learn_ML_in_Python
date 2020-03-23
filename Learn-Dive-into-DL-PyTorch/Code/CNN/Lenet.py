# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : boyuai
# @Editor  : sublime 3
#

import sys
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import time

# 通过Sequential类来实现LeNet模型
# net


class Flatten(torch.nn.Module):  # 展平操作

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Reshape(torch.nn.Module):  # 将图像大小重定型

    def forward(self, x):
        return x.view(-1, 1, 28, 28)  # (B x C x H x W)

net = torch.nn.Sequential(  # Lelet
    Reshape(),
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,
              padding=2),  # b*1*28*28  =>b*6*28*28
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # b*6*28*28  =>b*6*14*14
    nn.Conv2d(in_channels=6, out_channels=16,
              kernel_size=5),  # b*6*14*14  =>b*16*10*10
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # b*16*10*10  => b*16*5*5
    Flatten(),  # b*16*5*5   => b*400
    nn.Linear(in_features=16 * 5 * 5, out_features=120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

# 构造一个高和宽均为28的单通道数据样本，并逐层进行前向计算来查看每个层的输出形状
# print
X = torch.randn(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)


# 数据
#
def load_data_fashion_mnist(batch_size, resize=None, root='../../dataset/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(
    batch_size=batch_size, root='../../dataset/FashionMNIST')
print(len(train_iter))
