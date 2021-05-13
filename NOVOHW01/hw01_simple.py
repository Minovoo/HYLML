#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
==================================================
@project -> file ：LHYML -> hw01_org.PY
@author: Minovo
@time  : 2021/3/11 1:23 AM
@IDE   : PyCharm
@site   : 
@desc  : 
==================================================
"""

'''load train data train.csv'''
'''
全局禁止警告：
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
print('x' in np.arange(5))   #returns False, without Warning
逐行抑制警告.

import warnings
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    print('x' in np.arange(2))   #returns False, warning is suppressed

print('x' in np.arange(10))   #returns False, Throws FutureWarning
'''
import sys
import pandas as pd
import numpy as np
import warnings
import math
warnings.simplefilter(action='ignore', category=FutureWarning)
# print('x' in np.arange(5))   #returns False, without Warning

data = pd.read_csv('./trainsimple.csv', encoding='big5')
'''clear data '''

'''取需要的數值部分，將 'RAINFALL' 欄位全部補 0。
另外，如果要在 colab 重覆這段程式碼的執行，請從頭開始執行(把上面的都重新跑一次)，以避免跑出不是自己要的結果（若自己寫程式不會遇到，但 colab 重複跑這段會一直往下取資料。意即第一次取原本資料的第三欄之後的資料，第二次取第一次取的資料掉三欄之後的資料，...）。'''

data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

'''Extract Features  將原始 5760 * 5 的資料依照每個月分重組成 12 個 18 (features) * 480 (hours) 的資料。'''
'''Extract Features  3月3天4小时3特征將原始 36 * 3特征 的資料依照每個月分重組成 3 個 3 (features) * 12 (hours:3day*4hours) 的資料。'''

month_data = {}
'''3 month '''
for month in range(3):
    '''3feature * 3day * 4hours'''
    sample = np.empty([3, 12])
    '''3day'''
    for day in range(3):
        '''       day * 4hours,             3feature * (3day*month+day        '''
        sample[:, day * 4: (day+1) * 4] = raw_data[3 * (3 * month + day): 3 * (3*month + day+1), :]
    month_data[month] = sample

'''Extract Features2 3月3天4小时3特，每個月會有 12hrs，每 2 小時形成一個 data，每個月會有 12-2=10 個 data，故總資料數為 12-2=10 * 3month 筆，
而每筆 data 有 2 * 3 的 features (一小時 3 個 features * 2 小時)。
對應的 target 則有 （12-2） * 3 個(第 3 個小時的 PM2.5)
'''
'''         3month * 12hrs-2hrs, 3features * 2 hrs'''
x = np.empty([3 * 10, 3 * 2], dtype=float)
'''            y  3* 1month * (total 12hrs-2hrs)         '''
y = np.empty([3 * 10, 1], dtype=float)
for month in range(3):
    for day in range(3):
        for hour in range(4):
            if day == 2 and hour > 1:
                continue
            x[month * 10 + day * 4 +hour, :] = month_data[month][:,day * 4 +hour:day * 4 + hour + 2].reshape(1, -1)

            y[month * 10 + day * 4 + hour, 0] = month_data[month][2, day * 4 + hour + 2]

'''Normalize (1)'''
''' corr standard:  X* = (X - E(X)) / (var ** 0.5) , cov(X*, Y*) = corr(X,Y)'''
mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)

print('***')

for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
'''#**Split Training Data Into "train_set" and "validation_set"**'''
'''這部分是針對作業中 report 的第二題、第三題做的簡單示範，以生成比較中用來訓練的 train_set 和不會被放入訓練、
只是用來驗證的 validation_set。'''
print('****')
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]

# print(x_train_set)
# print(y_train_set)
# print(x_validation)
# print(y_validation)
print(len(x_train_set))
print(len(y_train_set))
print(len(x_validation))
print(len(y_validation))


'''   3 feature * 每2 days '''
dim = 3 * 2 + 1
w = np.zeros([dim, 1])
print('2*' * 50)
'''                          3month * 12-2hrs  '''
x = np.concatenate((np.ones([3 * 10, 1]), x), axis=1).astype(float)
learning_rate = 100
iter_time = 5
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 10 / 3)  # rmse
    # if t % 100 == 0:
    #     print(str(t) + ":" + str(loss))
    print('1000000000*' * 20)
    print(loss)
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim*1
    print('2000000000*' * 20)
    print(gradient)
    # print(y)
    # print(t)
    # print(gradient)
    adagrad += gradient ** 2
    print('3000000000*' * 20)
    print(adagrad)
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    print('4000000000*' * 20)
    print(w)
np.save('weightsimple.npy', w)


