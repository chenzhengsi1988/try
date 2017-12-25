#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:20:58 2017

@author: zsc
"""

import pymongo
from pymongo import MongoClient
client = MongoClient('localhost',27017)
#指定数据库名称
db = client.test_database
print(db)
for data in client.find():
    print data
#client = MongoClient('mongodb://localhost:27017')
# 连接数据库
# # client = MongoClient('192.168.30.252', 27017)
# # 获取现有数据库的名称
# client.database_names()
# # 将现有的cp到新的
# client.admin.command('copydb', fromdb='foobar', todb='foobar_new')
# # {u'ok': 1.0}
# client.database_names()
# # [u'local', u'wocao', u'foobar_new', u'foobar', u'cube_test_2016_04_26', u'mofangdb_2016_06_22', u'test', u'cube_test']
#
# # 在没有密码的前提下，从不通的mongod服务器上copy数据库
# # client.admin.command('copydb',fromdb='远程数据库的名称',todb='本地目标的数据库名称',fromhost='远程mongo的host地址')
# # 如果远程mongdb服务存在密码
# client.admin.authenticate('administrator', 'pwd')
# client.admin.command('copydb',
#                      fromdb='source_db_name',
#                      todb='target_db_name',
#                      fromhost='source.example.com')