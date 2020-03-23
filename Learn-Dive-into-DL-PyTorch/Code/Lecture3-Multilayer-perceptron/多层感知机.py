# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : boyuai
# @Editor  : sublime 3
#
"""

多层感知机从零开始实现

"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import sys

# print(torch.__version__)


# # tensor .detach() 返回和 x 的相同数据 tensor,而且这个新的tensor和原来的tensor是共用数据的，
# # 一者改变，另一者也会跟着改变，而且新分离得到的tensor的require s_grad = False, 即不可求导的。
# # （这一点其实 .data是一样的）


# def xyplot(x_vals, y_vals, name):
#     # d2l.set_figsize(figsize=(5, 2.5))
#     plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
#     plt.xlabel('x')
#     plt.ylabel(name + '(x)')
#     plt.show()  # 非notebook运行时加上以显示图像

# x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = x.relu()
# xyplot(x, y, 'relu')


# y.sum().backward()  # 注意.sum()为标量进行运算

# # xyplot(x, x.grad, 'grad of relu')

# y = x.sigmod()
# # xyplot(x,y,'sigmod')

# # sigmoid函数求导
# x.grad.zero_()
# y.sum().backward()
# print(x.grad)

# # tanh函数求导
# y = x.tanh()
# x.grad.zero_()
# y.sum().backward()

# 多层感知机从零开始实现
print("多层感知机从零开始实现")

# 获取数据


def load_data_fashion_mnist(batch_size, resize=None, root='../../dataset'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)  # Compose函数用于组合多个图像变换
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
path = '../../dataset'
train_iter, test_iter = load_data_fashion_mnist(batch_size, root=path)

# 定义参数模型
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(
    0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(
    0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

# 定义激活函数


def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

# 定义网络


def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

# 定义损失函数
loss = torch.nn.CrossEntropyLoss()

# 训练
print("开始训练")


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
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

num_epochs, lr = 5, 100.0
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
