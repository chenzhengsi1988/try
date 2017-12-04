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
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit([[0], [1],[2],[3]])


# print "enc.n_values_ is:",enc.n_values_
# print "enc.feature_indices_ is:",enc.feature_indices_
# print enc.transform([[0]]).toarray()
# print enc.transform([[2]]).toarray()
# print enc.transform([[1],[1],[1],[0],[2]]).toarray()
# 读取数据
data = pd.read_csv('~/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_training_dl.csv')
# 拆分数据
X_train, X_test, y_train1, y_test1 = train_test_split(
    data[['status',
 'location_speed',
 'httpcode_5m_200',
 'httpcode_5m_302',
 'httpcode_5m_404',
 'httpcode_5m_403',
 'httpcode_5m_500',
 'httpcode_30m_200',
 'httpcode_30m_302',
 'httpcode_30m_404',
 'httpcode_30m_403',
 'httpcode_30m_500',
 'httpcode_1d_200',
 'httpcode_1d_302',
 'httpcode_1d_404',
 'httpcode_1d_403',
 'httpcode_1d_500',
 'body_bytes_sent',
 'upstream_response_time',
 'httpcode_total_200',
 'httpcode_total_302',
 'httpcode_total_404',
 'httpcode_total_403',
 'httpcode_total_500',
 'request_time']].values, data["label"].values.reshape(-1, 1),random_state=42)
# one-hot 编码
# print(y_train1.shape)
# print(type(y_train1))
y_train1 = enc.transform(y_train1).toarray()
y_test1 = enc.transform(y_test1).toarray()
y_train= tf.convert_to_tensor(y_train1)
y_test = tf.convert_to_tensor(y_test1)
print(y_train.shape)
print(type(y_train))
# print(y_train.shape)
# 设置模型
learning_rate = 0.00001
training_epoch = 50
batch_size = 100
display_step = 1

n_samples = X_train.shape[0]
# print(n_samples)
n_features = 25
n_class = 4
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
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化所有变量
init = tf.initialize_all_variables()

aa = list()
bb = list()
# 训练模型
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epoch):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)

        for i in range(total_batch):
            batch_xs = X_train[i * batch_size: (i + 1) * batch_size]
            batch_ys = sess.run(y_train[i * batch_size: (i + 1) * batch_size])

            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / total_batch

        if (epoch + 1) % display_step == 0:
            print("Epoch:", "%04d" % (epoch + 1), "cost=", avg_cost)
            aa.append(epoch + 1)
            bb.append(avg_cost)
    print("Optimization Finished!")
    print("Testing Accuracy:", accuracy.eval({x: X_train, y: y_train.eval()}))

    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.plot(aa, bb)
    plt.show()
