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
import os

# 读取数据
data = pd.read_csv("~/ml/data/datatraining.txt")
# 拆分数据
X_train, X_test, y_train, y_test = train_test_split(
    data[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]].values, data["Occupancy"].values.reshape(-1, 1),
    random_state=42)
# one-hot 编码
# print(y_train.shape)
y_train = tf.concat([1 - y_train, y_train], 1)
y_test = tf.concat([1 - y_test, y_test], 1)
# print(y_train.shape)
# 设置模型
learning_rate = 0.001
training_epoch = 50
batch_size = 1000
display_step = 1

n_samples = X_train.shape[0]
# print(n_samples)
n_features = 5
n_class = 2
x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_class])

# 模型参数

W = tf.Variable(tf.zeros([n_features, n_class]))
b = tf.Variable(tf.zeros([n_class]))
# W = tf.Variable(tf.truncated_normal([n_features, n_class-1]))
# b = tf.Variable(tf.truncated_normal([n_class]))



# 计算预测值
pred = tf.nn.softmax(tf.matmul(x, W) + b)
# 计算损失值 使用相对熵计算损失值
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
# 定义优化器
# optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# 准确率
# correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化所有变量
init = tf.initialize_all_variables()

# aa=list()
# bb=list()
# 训练模型
with tf.Session() as sess:
    sess.run(init)

    # 定义模型保存对象
    saver = tf.train.Saver()

    tf.add_to_collection('x', x)
    tf.add_to_collection('pred', pred)

    for epoch in range(training_epoch):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)

        for i in range(total_batch):
            batch_xs = X_train[i * batch_size: (i + 1) * batch_size]
            batch_ys = sess.run(y_train[i * batch_size: (i + 1) * batch_size])

            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / total_batch

            #        if (epoch+1) % display_step == 0:
            #            print("Epoch:", "%04d" % (epoch+1), "cost=", avg_cost)
            # aa.append(epoch+1)
            # bb.append(avg_cost)
    print("Optimization Finished!")
    # 创建模型保存目录
    model_dir = "cx1"
    model_name = "ckp"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # 保存模型
    saver.save(sess, os.path.join(model_dir, model_name))

    print("保存模型成功！")


