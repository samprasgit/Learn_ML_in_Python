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


train_path = glob.glob('/content/drive/My Drive/Colab/街景字符识别/data/train/*.png')
train_path.sort()
train_json = json.load(
    open('/content/drive/My Drive/Colab/街景字符识别/data/train.json'))
train_label = [train_json[x]['label'] for x in train_json]

train_loader = torch.utils.data.DataLoader(SVHNDataset(train_path, train_label, transforms.Compose([
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
    batch_size=10,  # 每批样本个数
    shuffle=False,  # 是否打乱顺序
    num_workers=10,  # 读取的线程个数
)
