# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : boyuai

"""

线性回归模型pytorch简洁实现

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(1)
print(torch.__version__)
torch.set_default_tensor_type('torch.FloarTensor')\

import torch.utils.data as Data


# 生成数据集
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

features = torch.tensor(np.random.normal(
    0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size(), dtype=torch.float)


# 读取数据集
batch_size=10

# combine features and labels of dataset
dataset=Data.TensorDataset(features, labels)
# put dataset into Dataloader
data_iter=Data.Dataloader(
	dataset=dataset,
	batch_size=batch_size,
	shuffle=True,
	num_workers=2,
	)

for X, y in data_iter:
	print(X, '\n', y)
	break

# 定义模型
class LinearNet(nn.Module):
	def __inti__(self, n_feature):
		# call father function to init
		super(LinearNet, self).__init__()
		slef.linear=nn.Linear(n_feature, 1)

	def forward(slef, x):
		y=self.linear(x)
		return y

net=LinearNet(num_inputs)
print(net)

# 定义多层网络
"""
定义多层网络
# method one
net = nn.Sequential(
	nn.Linear(num_inputs,1)
	# other layers can be added here
	)

# method two
net = nn.Sequential()
net.add_module('linear',nn.Linear(num_inputs,1))

# method 3
from collection import OrderedDict
net = nn.Sequential(OrderedDict([
      ('linear',nn.Linear(num_inputs,1))
      # ......
      ]))

print(net)
print(net[0])

"""

# 初始化模型参数
from torch.nn import init

init.normal_(net[0].weight, mean=0.0, std=0.01)
inti.constant_(net[0].bias, val=0.0)

for param in net.parammeters():
	print(param)

# 定义损失函数
#
loss=nn.MSELoss()

# 定义优化函数
import torch.optim as optim
optimizer=optim.SGD(net.parammeters(), lr=0.03)
print(optimier)

# train
num_epochs=3
for epoch in range(1, num_epochs + 1):
	for X, y in data_iter:
		output=net(X)
		l=loss(output, y.view(-1, 1))
		optimizer.zero_grad()
		l.backward()
		optimizer.step()
	print('epoch %d,loss:%f' % (epoch, l.item()))

# result comparision
dense=ne[0]
print(true_w, dense.weight.data)
print(true_b, dense.bias.data)
