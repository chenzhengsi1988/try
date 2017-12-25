#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:20:58 2017

@author: zsc
"""

import pymongo
from pymongo import MongoClient

# client = MongoClient()

client = MongoClient('localhost', 27017)
print client
# 连接mongodb数据库
# client = MongoClient('mongodb://10.0.0.9:27017/')
# 指定数据库名称
db = client.Jikexueyuan
print db
# 获取非系统的集合
# db.collection_names(include_system_collections=False)
# 获取集合名
posts = db.test
print posts
for post in posts.find():
    print post
# 查找单个文档
print posts.find_one({'name':u'玉皇大帝'})