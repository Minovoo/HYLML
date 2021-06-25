#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
======================
@project -> file ：HYLML -> hw04_org.PY
@author: Minovo
@time  : 6/25/2021 11:44
@IDE   : PyCharm
@desc  : 
======================
"""


'''# Recurrent Neural Networks

本次作業是要讓同學接觸NLP當中一個簡單的task——句子分類(文本分類)

給定一個句子，判斷他有沒有惡意(負面標1，正面標0)

若有任何問題，歡迎來信至助教信箱ntu-ml-2020spring-ta@googlegroups.com

#%%'''

# from google.colab import drive
# drive.mount('/content/drive')
# path_prefix = 'drive/My Drive/Colab Notebooks/hw4 - Recurrent Neural Network'

path_prefix = './'

'''#%% md

### Download Dataset
有三個檔案，分別是training_label.txt、training_nolabel.txt、testing_data.txt

training_label.txt：有label的training data(句子配上0 or 1)

training_nolabel.txt：沒有label的training data(只有句子)，用來做semi-supervise learning

testing_data.txt：你要判斷testing data裡面的句子是0 or 1

#%%
'''

# %%

# !wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1dPHIl8ZnfDz_fxNd2ZeBYedTat2lfxcO' -O 'drive/My Drive/Colab Notebooks/hw8-RNN/data/training_label.txt'
# !wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1x1rJOX_ETqnOZjdMAbEE2pqIjRNa8xcc' -O 'drive/My Drive/Colab Notebooks/hw8-RNN/data/training_nolabel.txt'
# !wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=16CtnQwSDCob9xmm6EdHHR7PNFNiOrQ30' -O 'drive/My Drive/Colab Notebooks/hw8-RNN/data/testing_data.txt'

# !gdown --id '1lz0Wtwxsh5YCPdqQ3E3l_nbfJT1N13V8' --output data.zip
# !unzip data.zip
# !ls

# !ls 'drive/My Drive/Colab Notebooks/hw4 - Recurrent Neural Network/data'

# %%
# this is for filtering the warnings

import warnings

warnings.filterwarnings('ignore')
# %% md

### Utils

# %%
# utils.py
# 這個block用來先定義一些等等常用到的函式
import torch
import numpy as np
import pandas as np
import torch.optim as optim
import torch.nn.functional as F

def lo
