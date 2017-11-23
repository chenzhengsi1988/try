#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:20:58 2017

@author: zsc
"""

from __future__ import print_function, division
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.cross_validation import train_test_split
import random

# 读取数据
data = pd.read_csv("/Users/zsc/Desktop/datatraining.txt")
# 拆分数据
X_train, X_test, y_train, y_test = train_test_split(data[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]].values, data["Occupancy"].values.reshape(-1, 1), random_state=42)
# one-hot 编码
#print(y_train.shape)
y_train = tf.concat([1 - y_train, y_train],1)
y_test = tf.concat([1 - y_test, y_test],1)






# 创建会话
sess = tf.Session()

new_saver = tf.train.import_meta_graph('cx1/ckp.meta')
new_saver.restore(sess, 'cx1/ckp')

x = tf.get_collection('x')[0]
pred = tf.get_collection('pred')[0]

print("恢复模型成功！")

# 取出一条测试样本
idx = 1
record= X_test[idx]

# 根据模型计算结果
ret = sess.run(pred, feed_dict = {x : record.reshape(1,5)})

print("计算模型结果成功！")

# 显示测试结果
print("预测结果:")
print(ret)
print("实际结果:")
print(sess.run(y_test[idx]))



