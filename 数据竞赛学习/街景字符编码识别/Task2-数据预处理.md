<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [2.1图像读取](#21%E5%9B%BE%E5%83%8F%E8%AF%BB%E5%8F%96)
- [2.2图像数据扩增方法(Data Augementation)](#22%E5%9B%BE%E5%83%8F%E6%95%B0%E6%8D%AE%E6%89%A9%E5%A2%9E%E6%96%B9%E6%B3%95data-augementation)
    - [为神魔进行图像数据扩增](#%E4%B8%BA%E7%A5%9E%E9%AD%94%E8%BF%9B%E8%A1%8C%E5%9B%BE%E5%83%8F%E6%95%B0%E6%8D%AE%E6%89%A9%E5%A2%9E)
    - [数据扩增方法：](#%E6%95%B0%E6%8D%AE%E6%89%A9%E5%A2%9E%E6%96%B9%E6%B3%95)
    - [常用的数据扩增库](#%E5%B8%B8%E7%94%A8%E7%9A%84%E6%95%B0%E6%8D%AE%E6%89%A9%E5%A2%9E%E5%BA%93)
- [2.3Torch读取数据](#23torch%E8%AF%BB%E5%8F%96%E6%95%B0%E6%8D%AE)
- [参考](#%E5%8F%82%E8%80%83)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 2.1图像读取

- ### Pillow

  Pillow是Python图像处理函式库(PIL）的一个分支。Pillow提供了常见的图像读取和处理的操作，而且可以与ipython notebook无缝集成，是应用比较广泛的库。  

  Pillow的官方文档：https://pillow.readthedocs.io/en/stable/

- ### OpenCV

  OpenCV是一个跨平台的计算机视觉库，最早由Intel开源得来。OpenCV发展的非常早，拥有众多的计算机视觉、数字图像处理和机器视觉等功能。OpenCV在功能上比Pillow更加强大很多，学习成本也高很多。 

  OpenCV包含了众多的图像处理的功能，OpenCV包含了你能想得到的只要与图像相关的操作。此外OpenCV还内置了很多的图像特征处理算法，如关键点检测、边缘检测和直线检测等

  OpenCV官网：https://opencv.org/     

  OpenCV Github：https://github.com/opencv/opencv      

  OpenCV 扩展算法库：https://github.com/opencv/opencv_contrib

## 2.2图像数据扩增方法(Data Augementation)

#### 为神魔进行图像数据扩增

在深度学习中数据扩增方法非常重要，数据扩增可以增加训练集的样本，同时也可以有效缓解模型过拟合的情况，也可以给模型带来的更强的泛化能力。

#### 数据扩增方法：

数据扩增方法有很多：从颜色空间、尺度空间到样本空间，同时根据不同任务数据扩增都有相应的区别

- 图像分类，数据扩增一般不会改变标签
- 物体检测，数据扩增会改变物体坐标位置
- 图像分割，数据扩增会改变像素标签

在常见的数据扩增方法中，一般会从图像颜色、尺寸、形态、空间和像素等角度进行变换。当然不同的数据扩增方法可以自由进行组合，得到更加丰富的数据扩增方法。         

#### 常用的数据扩增库   

- [**torchvision** ](https://github.com/pytorch/vision)   

  pytorch官方提供的数据扩增库，提供了基本的数据数据扩增方法，可以无缝与torch进行集成；但数据扩增方法种类较少，且速度中等

- [imgaug](https://github.com/aleju/imgaug   )

  imgaug是常用的第三方数据扩增库，提供了多样的数据扩增方法，且组合起来非常方便，速度较快

- [albumentations](https://albumentations.readthedocs.io     )

  是常用的第三方数据扩增库，提供了多样的数据扩增方法，对图像分类、语义分割、物体检测和关键点检测都支持，速度较快

以`torchvision`为例，常见的数据扩增方法包括：

- 裁剪--Crop 

  中心裁剪：transforms.CenterCrop

  随机裁剪：transforms.RandomCrop

  随机长宽比裁剪：transforms.RandomResizedCrop

  上下左右中心裁剪：transforms.FiveCrop

  上下左右中心裁剪后翻转：transforms.TenCrop

- 翻转和旋转--Flip and Rotation

  依概率p水平旋转：transforms.RandomHorizontalFlip(p=0.5)

  依概率p垂直旋转：transforms.RandomVertialFlip(p=0.5)

  随机旋转：transforms.RandomRotation

- 图像变换

  Resize ：transforms.Resize

  标准化：transforms.Normalize

  转为tensor并归一化至[0-1]：transforms.ToTensor

  修改亮度、对比度和饱和度:transforms.ColorJitter 

  转灰度图:transforms.Grayscale 

  线性变换:transforms.LinearTransformation

  仿射变换:transforms.RandomAffine
  依概率 p 转为灰度图:transforms.RandomGrayscale(p=) 

  将数据转换为 PILImage:transforms.ToPILImage 

- 对transforms操作，是数据增强更加灵活

   从给定的一系列 transforms 中选一个进行操作:transforms.RandomChoice(transforms)

  给一个 transform 加上概率，依概率进行操作:transforms.RandomApply(transforms, p=0.5)

  将 transforms 中的操作随机打乱: transforms.RandomOrder

![数据扩增示例](Task2-数据预处理.assets/数据扩增示例.png)

## 2.3Torch读取数据

在Pytorch中数据是通过Dataset进行封装，并通过DataLoder进行并行读取。所以我们只需要重载一下数据读取的逻辑就可以完成数据的读取

- Dataset：对数据集的封装，提供索引方式的对数据样本进行读取     `__getitem__`中进行重载
- DataLoder：对Dataset进行封装，提供批量读取的迭代读取

**定义SVHNDataset数据集**

```python
# 定义SVHNDataset
import os
import sys
import glob
import shutil
import json
import cv2

from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


class SVHNDataset(Dataset):

    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # 原始SVHN中类别10为数字0
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (5 - len(lbl)) * [10]

        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)

```

```python 
train_path=glob.glob('/content/drive/My Drive/Colab/街景字符识别/data/train/*.png')
train_path.sort() 
train_json=json.load(open('/content/drive/My Drive/Colab/街景字符识别/data/train.json'))
train_label=[train_json[x]['label'] for x in train_json]

data = SVHNDataset(train_path, train_label,
          transforms.Compose([
              # 缩放到固定尺寸
              transforms.Resize((64, 128)),

              # 随机颜色变换
              transforms.ColorJitter(0.2, 0.2, 0.2),

              # 加入随机旋转
              transforms.RandomRotation(5),

            #   将图片转换为pytorch 的tesntor
              transforms.ToTensor(),

            #   对图像像素进行归一化
              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]))
```

单个数据样本格式为：

```python
data[0][0].shape
```

```
torch.Size([3, 64, 128])
```

**torch.utils.data.DataLoader实现数据集加载**

```python
train_path = glob.glob('/content/drive/My Drive/Colab/街景字符识别/data/train/*.png')
train_path.sort()
train_json = json.load(
    open('/content/drive/My Drive/Colab/街景字符识别/data/train.json'))
train_label = [train_json[x]['label'] for x in train_json]

train_loader = torch.utils.data.DataLoader(
    SVHNDataset(
        train_path,
        train_label,
        transforms.Compose([
            #   缩放到固定尺寸
            transforms.Resize((64, 128)),
            #   随机颜色变换
            transforms.ColorJitter(0.3, 0.3, 0.2),
            #   加入随机旋转
            transforms.RandomRotation(5),
            #   将图片转换为pytorch的tensor
            transforms.ToTensor(),
            #   对图像元素进行归一化
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])),
    batch_size=40,  # 每批样本个数
    shuffle=False,  # 是否打乱顺序
    num_workers=10,  # 读取的线程个数
)
```

注:DataLoader返回的是一个可迭代对象，我们可以使用迭代器分次获取数据

在加入DataLoder后，数据按照批次获取，每批次调用Dataset读取单个样本进行拼接。此时data的格式为

```
torch.Size([40, 3, 64, 128]), torch.Size([10, 6])
```

前者为图像文件，为batchsize * chanel * height * width次序；后者为字符标签

## 参考

[https://github.com/datawhalechina/team-learning/blob/master/03%20%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5%EF%BC%88%E8%A1%97%E6%99%AF%E5%AD%97%E7%AC%A6%E7%BC%96%E7%A0%81%E8%AF%86%E5%88%AB%EF%BC%89/Datawhale%20%E9%9B%B6%E5%9F%BA%E7%A1%80%E5%85%A5%E9%97%A8CV%20-%20Task%2002%20%E6%95%B0%E6%8D%AE%E8%AF%BB%E5%8F%96%E4%B8%8E%E6%95%B0%E6%8D%AE%E6%89%A9%E5%A2%9E.md](https://github.com/datawhalechina/team-learning/blob/master/03 计算机视觉/计算机视觉实践（街景字符编码识别）/Datawhale 零基础入门CV - Task 02 数据读取与数据扩增.md)

