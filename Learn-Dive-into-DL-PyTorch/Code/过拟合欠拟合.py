# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : boyuai
# @Editor  : sublime 3
#

"""
过拟合 欠拟合及其解决方案
"""
################
# 1.多项式模拟实验
################

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys


# import d2lzh1981 as d2l
print(torch.__version__)

# 初始化模型参数
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))
poly_features = torch.cat(
    (features, torch.pow(features, 2), torch.pow(features, 3)), 1)
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01,
                                        size=labels.size()), dtype=torch.float)


# 定义训练和测试模型
num_epochs, loss = 100, torch.nn.MSELoss()


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


def fit_and_plot(train_features, test_features, train_labels, test_labels):
    # 初始化网络模型
    net = torch.nn.Linear(train_features.shape[-1], 1)
    # 通过Linear文档可知，pytorch已经将参数初始化了，所以我们这里就不手动初始化了

    # 设置批量大小
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(
        train_features, train_labels)      # 设置数据集
    train_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True)  # 设置获取数据方式

    # 设置优化函数，使用的是随机梯度下降优化
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:                                                 # 取一个批量的数据
            # 输入到网络中计算输出，并和标签比较求得损失函数
            l = loss(net(X), y.view(-1, 1))
            # 梯度清零，防止梯度累加干扰优化
            optimizer.zero_grad()
            l.backward()                                                        # 求梯度
            # 迭代优化函数，进行参数优化
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        # 将训练损失保存到train_ls中
        train_ls.append(loss(net(train_features), train_labels).item())
        # 将测试损失保存到test_ls中
        test_ls.append(loss(net(test_features), test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,
          '\nbias:', net.bias.data)

# 三阶多项式函数拟合（正常）
fit_and_plot(poly_features[:n_train, :], poly_features[
             n_train:, :], labels[:n_train], labels[n_train:])

# 线性函数你和拟合（欠拟合）
fit_and_plot(features[:n_train, :], features[n_train:, :],
             labels[:n_train], labels[n_train:])
# 训练样本不足（过拟合）
fit_and_plot(poly_features[0:2, :], poly_features[
             n_train:, :], labels[0:2], labels[n_train:])


# 高维线性回归从零开始实现
# 权重衰减应对过拟合

# 初始化模型参数
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01,
                                        size=labels.size()), dtype=torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 定义参数初始化函数，初始化模型参数并且附上梯度


def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 定义L2惩罚项


def l2_penalty(w):
    return (w**2).sum() / 2

# 定义训练和测试
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 添加了L2范数惩罚项
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b),
                             train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b),
                            test_labels).mean().item())
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().item())

# 观察过拟合
fit_and_plot(lambd=0)

# 使用权重衰减
fit_and_plot(lambd=3)

# 使用torch简洁实现权重衰减


def fit_and_plot_pytorch(wd):
    # 对权重参数衰减。权重名称一般是以weight结尾
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    optimizer_w = torch.optim.SGD(
        params=[net.weight], lr=lr, weight_decay=wd)  # 对权重参数衰减
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)  # 不对偏差参数衰减

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            l.backward()

            # 对两个optimizer实例分别调用step函数，从而分别更新权重和偏差
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.weight.data.norm().item())

# 观察过拟合
fit_and_plot_pytorch(0)
# 权重衰减解决过拟合
fit_and_plot_pytorch(3)
