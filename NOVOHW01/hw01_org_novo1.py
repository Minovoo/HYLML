#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
======================
@project -> file ：HYLML -> hw01_org_novo1.PY
@author: Minovo
@time  : 4/1/2021 09:59
@IDE   : PyCharm
@desc  : 
======================
"""

import sys
import sys
import pandas as pd
import numpy as np
import csv
import math

# load train data .csv
data = pd.read_csv('./train.csv', encoding='big5')

# clear data
'''取需要的數值部分，將 'RAINFALL' 欄位全部補 0。
另外，如果要在 colab 重覆這段程式碼的執行，請從頭開始執行(把上面的都重新跑一次)，以避免跑出不是自己要的結果（若自己寫程式不會遇到，但 colab 重複跑這段會一直往下取資料。意即第一次取原本資料的第三欄之後的資料，第二次取第一次取的資料掉三欄之後的資料，...）。'''

data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

'''Extract Features  將原始 5760 * 18 的資料依照每個月分重組成 12 個 18 (features) * 480 (hours)[12 month 20day 20 hrs] 的資料。'''
'''Extract Features  3月4天24小时5特征將原始 288 * 5特征 的資料依照每個月分重組成 3 個 5 (features) * 96 (hours) 的資料。'''

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1)]
        if day == 0:
            print(sample)
    month_data[month] = sample
'''Extract Features2 每個月會有 480hrs，每 9 小時形成一個 data，每個月會有 471 個 data生成，故總資料數為 471 * 12 筆，
而每筆 data 有 9 * 18 的 features (一小時 18 個 features * 9 小時)。
對應的 target 則有 471 * 12 個(第 10 個小時的 PM2.5)
'''
x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24+hour + 9].reshape(1, -1)
            # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]
            # value
# np.savetxt('dbempty.txt', x[470:490, :], delimiter=',', fmt='%.2f')
'''Normalize (1)'''
''' corr standard:  X* = (X - E(X)) / (var ** 0.5) , cov(X*, Y*) = corr(X,Y)'''
mean_x = np.mean(x, axis=0) # 18 * 9
std_x = np.std(x, axis=0) # 18 * 9
for i in range(len(x)): # 12 * 471
    for j in range(len(x[0])): # 18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
print(len(x), len(y))
'''#**Split Training Data Into "train_set" and "validation_set"**'''
'''這部分是針對作業中 report 的第二題、第三題做的簡單示範，以生成比較中用來訓練的 train_set 和不會被放入訓練、
只是用來驗證的 validation_set。'''
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]
# f = open('dbempt.csv', 'w', encoding='big5')
# csvwr = csv.writer(f)
# csvwr.writerows(x[0:50,:])
# f.close()
# np.savetxt('dbempty.txt', x[0:10, :], delimiter=',', fmt='%.2f')

'''Training
(和上圖不同處: 下面的 code 採用 Root Mean Square Error RMSE)

因為常數項的存在，所以 dimension (dim) 需要多加一欄；eps 項是避免 adagrad 的分母為 0 而加的極小數值。

每一個 dimension (dim) 會對應到各自的 gradient, weight (w)，透過一次次的 iteration (iter_time) 學習。'''
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
leaning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12) # rmse
    if t % 100 == 0:
        print(str(t) + ':'+ str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) # dim * 1
    adagrad += gradient ** 2
    w = w - leaning_rate * gradient / np.sqrt(adagrad + eps)
    np.save('weight_novo1.npy', w)
# ![alt text](https://drive.google.com/uc?id=1165ETzZyE6HStqKvgR0gKrJwgFLK6-CW)

# 載入 test data，並且以相似於訓練資料預先處理和特徵萃取的方式處理，使 test data 形成 240 個維度為 18 * 9 + 1 的資料。

# testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
testdata = pd.read_csv('./test.csv', header=None, encoding='big5')
testdata = testdata.iloc[:, 2:]
testdata[testdata == 'NR'] = 0
testdata = testdata.to_numpy()
'''20day * 12 month, 18feature * 9hrs/day'''
test_x = np.empty([240, 18 * 9],dtype=float)
for i in range(240):
    test_x[i, :] = testdata[18 * i: 18 * (i + 1),:].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
'''new test_x is 240 day , 18 * 9hrs+ones1'''
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)
# **Prediction**
# 說明圖同上
#
# ![alt text](https://drive.google.com/uc?id=1165ETzZyE6HStqKvgR0gKrJwgFLK6-CW)
#
# 有了 weight 和測試資料即可預測 target。

w = np.load('weight_novo1.npy')
ans_y = np.dot(test_x, w)
with open('submit_novo1.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)