#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
======================
@project -> file ：HYLML -> hw02_org.PY
@author: Minovo
@time  : 4/21/2021 09:38
@IDE   : PyCharm
@desc  : 
======================
"""
import numpy as np
np.random.seed(0)
# 1 path
hw02_path = 'D:/Workpy/HYLML/NOVOHW02'
X_train_fpath = hw02_path + '/X_train'
Y_train_fpath = hw02_path + '/Y_train'
X_test_fpath = hw02_path + '/X_test'
output_fpath = hw02_path + '/output2_{}.csv'

# 2 data
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)


# 3-1 normalize
def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    if specified_column == None:
        specified_column = np.array(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)
    X[:,specified_column] = (X[:,specified_column] - X_mean) / (X_std + 1e-8)
    return X, X_mean, X_std


# 3-2 split
def _train_dev_split(X, Y, dev_ratio = 0.25):
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


# 3-3 Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train=True)
X_test, _, _ = _normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)

# 3-4 split data
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)


# 3-5 size and dimension
train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set:{}'.format(train_size))
print('Size of development set:{}'.format(dev_size))
print('Size of test set:{}'.format(test_size))
print('Dimension of data:{}'.format(data_dim))


# 3-6 shuffle
def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


# 3-7 sigmoid
def _sigmoid(z):
    return np.clip(1/(1.0 + np.exp(-z)), 1e-8, 1-(1e-8))


# 3-8 f
def _f(X, w, b):
    return _sigmoid(np.matmul(X, w) + b)


# 3-9 predict
def _predict(X, w, b):
    return np.round(_f(X, w, b)).astype(np.int)


# 3-10 accuracy
def _accuracy(Y_pred, Y_label):
    acc = 1 - np.mean(np.abs(Y_pred, Y_label))
    return acc

# 3-11 cross entropy loss
def _cross_entropy_loss(y_pred, Y_label):
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1-Y_label), np.log(1-y_pred))
    return cross_entropy

# 3-12 gradient
def _gradient(X, Y_label, w, b):
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

w = np.zeros((data_dim,))
b = np.zeros((1,))

max_iter = 10
batch_size = 8
learning_rate = 0.2

train_loss = []
dev_loss = []
train_acc = []
dev_acc = []
step = 1
for epoch in range(max_iter):
    X_train, Y_train = _shuffle(X_train, Y_train)
    for idx in range(int(np.floor(train_size/batch_size))):
        X = X_train[idx * batch_size: (idx + 1) * batch_size]
        Y = Y_train[idx * batch_size: (idx + 1) * batch_size]
        w_grad, b_grad = _gradient(X, Y, w, b)
        w = w - learning_rate/np.sqrt(step) * w_grad
        b = b - learning_rate/np.sqrt(step) * w_grad
        step += 1
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)
print('Training loss:{}'.format(train_loss[-1]))
print('Development loss:{}'.format(dev_loss[-1]))
print('Training accuracy:{}'.format(train_acc[-1]))
print('Development accuracy:{}'.format(dev_acc[-1]))


import matplotlib.pyplot as plt
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()


predictions = _predict(X_test, w, b)
with open(output_fpath.format('logistic'), 'w') as f:
    f.write('id, label\n')
    for i, label in enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])







#
#
#
# import numpy as np
#
# # 确保生产相同的随机数
# np.random.seed(0)
# # 添加文件路径
# X_train_fpath = 'D:/Workpy/HYLML/NOVOHW02/X_train'
# Y_train_fpath = 'D:/Workpy/HYLML/NOVOHW02/Y_train'
# X_test_fpath = 'D:/Workpy/HYLML/NOVOHW02/X_test'
# output_fpath = 'D:/Workpy/HYLML/NOVOHW02/output_{}.csv'  # 用于测试集的预测输出
#
# -# Parse csv files to numpy array 加载数据，直接导入已经处理好的数据X_train,Y_train,X_test
# with open(X_train_fpath) as f:
#     next(f)
#     X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
# with open(Y_train_fpath) as f:
#     next(f)
#     Y_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
# with open(X_test_fpath) as f:
#     next(f)
#     X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
#
#
# # 标准化
# def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
#     # This function normalizes specific columns of X.
#     # The mean and standard variance of training data will be reused when processing testing data.
#     #
#     # Arguments:
#     #     X: data to be processed
#     #     train: 'True' when processing training data, 'False' for testing data
#     #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
#     #         will be normalized.
#     #     X_mean: mean value of training data, used when train = 'False'
#     #     X_std: standard deviation of training data, used when train = 'False'
#     # Outputs:
#     #     X: normalized data
#     #     X_mean: computed mean value of training data
#     #     X_std: computed standard deviation of training data
#     # 每列数据归一化 specified_column == None
#     if specified_column == None:
#         specified_column = np.arange(X.shape[1])
#     if train:
#         X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
#         X_std = np.std(X[:, specified_column], 0).reshape(1, -1)
#     X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
#     return X, X_mean, X_std
#
#
# def _train_dev_split(X, Y, dev_ratio=0.25):
#     # This function spilts data into training set and development set.
#     train_size = int(len(X) * (1 - dev_ratio))
#     return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]
#
#
# # Normalize training and testing data
# X_train, X_mean, X_std = _normalize(X_train, train=True)
# X_test, _, _ = _normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)
#
# # Split data into training set and development set
# dev_ratio = 0.1
# X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)
#
# train_size = X_train.shape[0]
# dev_size = X_dev.shape[0]
# test_size = X_test.shape[0]
# data_dim = X_train.shape[1]
# print('Size of training set: {}'.format(train_size))
# print('Size of development set: {}'.format(dev_size))
# print('Size of testing set: {}'.format(test_size))
# print('Dimension of data: {}'.format(data_dim))
#
#
# # %% md
#
#
# ###Some Useful Functions
#
# # Some functions that will be repeatedly used when iteratively updating the parameters.
# #
# # 這幾個函數可能會在訓練迴圈中被重複使用到。
#
# # %%
#
# def _shuffle(X, Y):
#     # This function shuffles two equal-length list/array, X and Y, together.
#     randomize = np.arange(len(X))
#     np.random.shuffle(randomize)
#     return (X[randomize], Y[randomize])
#
#
# def _sigmoid(z):
#     # Sigmoid function can be used to calculate probability.
#     # To avoid overflow, minimum/maximum output value is set.
#     return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))
#
#
# def _f(X, w, b):
#     # This is the logistic regression function, parameterized by w and b
#     #
#     # Arguements:
#     #     X: input data, shape = [batch_size, data_dimension]
#     #     w: weight vector, shape = [data_dimension, ]
#     #     b: bias, scalar
#     # Output:
#     #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
#     return _sigmoid(np.matmul(X, w) + b)
#
#
# def _predict(X, w, b):
#     # This function returns a truth value prediction for each row of X
#     # by rounding the result of logistic regression function.
#     return np.round(_f(X, w, b)).astype(np.int)
#
#
# def _accuracy(Y_pred, Y_label):
#     # This function calculates prediction accuracy
#     acc = 1 - np.mean(np.abs(Y_pred - Y_label))
#     return acc
#
#
# # %% md
#
# # Functions about gradient and loss
#
# # Please refers to [Prof. Lee's lecture slides](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/Logistic%20Regression%20(v3).pdf)(p.12) for the formula of gradient and loss computation.
# #
# # 請參考[李宏毅老師上課投影片](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/Logistic%20Regression%20(v3).pdf)第 12 頁的梯度及損失函數計算公式。
#
# # %%
#
#
# def _cross_entropy_loss(y_pred, Y_label):
#     # This function computes the cross entropy.
#     #
#     # Arguements:
#     #     y_pred: probabilistic predictions, float vector
#     #     Y_label: ground truth labels, bool vector
#     # Output:
#     #     cross entropy, scalar
#     cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
#     return cross_entropy
#
#
# def _gradient(X, Y_label, w, b):
#     # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
#     y_pred = _f(X, w, b)
#     pred_error = Y_label - y_pred
#     w_grad = -np.sum(pred_error * X.T, 1)
#     b_grad = -np.sum(pred_error)
#     return w_grad, b_grad
#
#
# # %% md
#
# ### Training
#
# # Everything is prepared, let's start training!
# #
# # Mini-batch gradient descent is used here, in which training data are split into several mini-batches and each batch is fed into the model sequentially for losses and gradients computation. Weights and bias are updated on a mini-batch basis.
# #
# # Once we have gone through the whole training set,  the data have to be re-shuffled and mini-batch gradient desent has to be run on it again. We repeat such process until max number of iterations is reached.
# #
# #
# # 我們使用小批次梯度下降法來訓練。訓練資料被分為許多小批次，針對每一個小批次，我們分別計算其梯度以及損失，並根據該批次來更新模型的參數。當一次迴圈完成，也就是整個訓練集的所有小批次都被使用過一次以後，我們將所有訓練資料打散並且重新分成新的小批次，進行下一個迴圈，直到事先設定的迴圈數量達成為止。
#
# # %%
#
# # Zero initialization for weights ans bias
# w = np.zeros((data_dim,))
# b = np.zeros((1,))
#
# # Some parameters for training
# max_iter = 10
# batch_size = 8
# learning_rate = 0.2
#
# # Keep the loss and accuracy at every iteration for plotting
# train_loss = []
# dev_loss = []
# train_acc = []
# dev_acc = []
#
# # Calcuate the number of parameter updates
# step = 1
# # Iterative training
# for epoch in range(max_iter):
#     # Random shuffle at the begging of each epoch
#     X_train, Y_train = _shuffle(X_train, Y_train)
#
#     # Mini-batch training
#     for idx in range(int(np.floor(train_size / batch_size))):
#         X = X_train[idx * batch_size:(idx + 1) * batch_size]
#         Y = Y_train[idx * batch_size:(idx + 1) * batch_size]
#
#
#
#
#         # Compute the gradient
#         w_grad, b_grad = _gradient(X, Y, w, b)
#
#         # gradient descent update
#         # learning rate decay with time
#         w = w - learning_rate / np.sqrt(step) * w_grad
#         b = b - learning_rate / np.sqrt(step) * b_grad
#
#         step = step + 1
#
#     # Compute loss and accuracy of training set and development set
#     y_train_pred = _f(X_train, w, b)
#     Y_train_pred = np.round(y_train_pred)
#     train_acc.append(_accuracy(Y_train_pred, Y_train))
#     train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)
#
#     y_dev_pred = _f(X_dev, w, b)
#     Y_dev_pred = np.round(y_dev_pred)
#     dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
#     dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)
#
# print('Training loss: {}'.format(train_loss[-1]))
# print('Development loss: {}'.format(dev_loss[-1]))
# print('Training accuracy: {}'.format(train_acc[-1]))
# print('Development accuracy: {}'.format(dev_acc[-1]))
#
# # #%% md
#
# ### Plotting Loss and accuracy curve
#
# # %%
#
# import matplotlib.pyplot as plt
#
# # Loss curve
# plt.plot(train_loss)
# plt.plot(dev_loss)
# plt.title('Loss')
# plt.legend(['train', 'dev'])
# plt.savefig('loss.png')
# plt.show()
#
# # Accuracy curve
# plt.plot(train_acc)
# plt.plot(dev_acc)
# plt.title('Accuracy')
# plt.legend(['train', 'dev'])
# plt.savefig('acc.png')
# plt.show()
#
# # %% md
#
# ###Predicting testing labels
#
# # Predictions are saved to *output_logistic.csv*.
# #
# # 預測測試集的資料標籤並且存在 *output_logistic.csv* 中。
#
# # %%
#
# # Predict testing labels
# predictions = _predict(X_test, w, b)
# with open(output_fpath.format('logistic'), 'w') as f:
#     f.write('id,label\n')
#     for i, label in enumerate(predictions):
#         f.write('{},{}\n'.format(i, label))
#
# # Print out the most significant weights
# ind = np.argsort(np.abs(w))[::-1]
# with open(X_test_fpath) as f:
#     content = f.readline().strip('\n').split(',')
# features = np.array(content)
# for i in ind[0:10]:
#     print(features[i], w[i])

# %% md

# Porbabilistic generative model

# In this section we will discuss a generative approach to binary classification. Again, we will not go through the formulation detailedly. Please find [Prof. Lee's lecture](https://www.youtube.com/watch?v=fZAZUYEeIMg) if you are interested in it.
#
# 接者我們將實作基於 generative model 的二元分類器，理論細節請參考[李宏毅老師的教學影片](https://www.youtube.com/watch?v=fZAZUYEeIMg)。
#
# ### Preparing Data
#
# Training and testing data is loaded and normalized as in logistic regression. However, since LDA is a deterministic algorithm, there is no need to build a development set.
#
# 訓練集與測試集的處理方法跟 logistic regression 一模一樣，然而因為 generative model 有可解析的最佳解，因此不必使用到 development set。

# %%

# # Parse csv files to numpy array
# with open(X_train_fpath) as f:
#     next(f)
#     X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
# with open(Y_train_fpath) as f:
#     next(f)
#     Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
# with open(X_test_fpath) as f:
#     next(f)
#     X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
#
# # Normalize training and testing data
# X_train, X_mean, X_std = _normalize(X_train, train = True)
# X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)
#
#
# #%% md
#
# ### Mean and Covariance
#
# # In generative model, in-class mean and covariance are needed.
# #
# # 在 generative model 中，我們需要分別計算兩個類別內的資料平均與共變異。
#
# #%%
#
# # Compute in-class mean
# X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
# X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])
#
# mean_0 = np.mean(X_train_0, axis = 0)
# mean_1 = np.mean(X_train_1, axis = 0)
#
# # Compute in-class covariance
# cov_0 = np.zeros((data_dim, data_dim))
# cov_1 = np.zeros((data_dim, data_dim))
#
# for x in X_train_0:
#     cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
# for x in X_train_1:
#     cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]
#
# # Shared covariance is taken as a weighted average of individual in-class covariance.
# cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])
#
# #%% md
#
# ### Computing weights and bias
#
# # Directly compute weights and bias from in-class mean and shared variance. [Prof. Lee's lecture slides](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/Classification%20(v3).pdf)(p.33) gives a concise explanation.
# #
# # 權重矩陣與偏差向量可以直接被計算出來，算法可以參考[李宏毅老師教學投影片](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/Classification%20(v3).pdf)第 33 頁。
#
# #%%
#
# # Compute inverse of covariance matrix.
# # Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
# # Via SVD decomposition, one can get matrix inverse efficiently and accurately.
# u, s, v = np.linalg.svd(cov, full_matrices=False)
# inv = np.matmul(v.T * 1 / s, u.T)
#
# # Directly compute weights and bias
# w = np.dot(inv, mean_0 - mean_1)
# b = (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))\
#     + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])
#
# # Compute accuracy on training set
# Y_train_pred = 1 - _predict(X_train, w, b)
# print('Training accuracy: {}'.format(_accuracy(Y_train_pred, Y_train)))
# #%% md
#
# ###Predicting testing labels
#
# # Predictions are saved to *output_generative.csv*.
# #
# # 預測測試集的資料標籤並且存在 *output_generative.csv* 中。
#
# #%%
#
# # Predict testing labels
# predictions = 1 - _predict(X_test, w, b)
# with open(output_fpath.format('generative'), 'w') as f:
#     f.write('id,label\n')
#     for i, label in  enumerate(predictions):
#         f.write('{},{}\n'.format(i, label))
#
# # Print out the most significant weights
# ind = np.argsort(np.abs(w))[::-1]
# with open(X_test_fpath) as f:
#     content = f.readline().strip('\n').split(',')
# features = np.array(content)
# for i in ind[0:10]:
#     print(features[i], w[i])
