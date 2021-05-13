#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
======================
@project -> file ：HYLML -> hw03_org.PY
@author: Minovo
@time  : 5/7/2021 09:58
@IDE   : PyCharm
@desc  : 
======================
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
def readfile(path, label):
    # label 是一个boolean variable， 代表需不需要返回值y值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
            y[i] = int(file.split('_')[0])
    if label:
        return x, y
    else:
        return x


# 分别将 training set , validation set, testing set 用readfile函式读进来
workspace_dir = 'D:/Workpy/HYLML/NOVOHW03/food-11'
print('reading data')
train_x, train_y = readfile(os.path.join(workspace_dir, 'training'), True)
print('Size of training data = {}'.format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, 'validation'), True)
print('Size of validation data = {}'.format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, 'testing'), False)
print('size of Testing data = {}'.format(len(test_x)))

# Dataset 在pytorch中欧冠， 我们可以利用torch.utils.data
# 的dataset及dataloader来‘包装’data,使后续的training
# testing更为方便。
# dataset 需要overload两个函数：\_\_len\_\_ 及
# \_\_getitem\_\_\_\_len\_\_ 必須要回傳 dataset 的大小，
# 而 \_\_getitem\_\_ 則定義了當程式利用 [ ] 取值時，
# dataset 應該要怎麼回傳資料。
# 實際上我們並不會直接使用到這兩個函數，但是使用 DataLoader
# 在 enumerate Dataset 時會使用到，沒有實做的話會在程式運行
# 階段出現 error。

# training 时做data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), # 随机将图片水平翻转
    transforms.RandomRotation(), # 随机旋转图片
    transforms.ToTensor(), #将图片转成Tensor, 并把数值normalize到[0,1](data normalization)
])

