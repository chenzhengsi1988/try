#-*-coding:utf8-*-
import pymongo

import json
import pandas as pd
from pandas import Series, DataFrame
from sklearn import preprocessing
import numpy as np

import sklearn
import random

tweets = []
connection = pymongo.MongoClient()
print connection
tdb = connection.Jikexueyuan
print tdb
post_info = tdb.yufu
post_info2 = tdb.yufu2
print post_info

# file=open("/Users/zsc/Desktop/testdata.txt", 'r')
# for line in file :
#     # tweets.append(json.loads(line))
#     # a={"date":"Thu Nov 16 18:04:00 CST 2017"}
#     # a={"date":"Thu Nov 16 18:04:00 CST 2017","operation_type":"download","httpcode_1d":{"200":4087,"500":255,"302":192,"403":231,"404":252},"body_bytes_sent":35114,"http_authorization":"-","http_host":"crm.xiaoshouyi.com","tid":"166912","http_user_agent":{"OperatingSystemName":"Windows 10","BrowserVersion":"61","Browser":"Chrome","DeviceName":"Computer"},"uid":"10000","request_time":35114,"http_content_type":"-","host_location":{"city":"北京","latitude":39.9289,"longitude":116.3883},"operation_obj":"txt2","remote_addr":"58.19.32.182","host_ip":"223.223.204.4","method":"GET","http_version":"HTTP/1.1","time_local":1.51082664E9,"label":0,"uri":"/json/crm_407/listNew.action","httpcode_total":{"200":4864,"500":312,"302":240,"403":273,"404":312},"httpcode_0_5h":{"200":121,"500":9,"302":3,"403":18,"404":6},"http_referer":"https://crm.xiaoshouyi.com/get.action","http_x_forwarded_for":"-","upstream_response_time":0.875,"location":{"distance":683.1672939399175,"city":"武汉","latitude":30.5801,"speed":500,"longitude":114.2734},"httpcode_5m":{"200":7},"v3":"\"v3\"","status":"200"}
#     #
#     # print type(a)
#     # post_info.insert(a)
#     cc=line
#     cr=cc.replace("httpcode_0.5h","httpcode_30_m")
#     print cr
#     a=json.loads(cr)
#     # print type(a)
#     post_info.insert(a)
    # print type(a)

# file.close
# jike = {'name':u'极客', 'age':'5', 'skill': 'Python'}
# god = {'name': u'玉皇大帝', 'age': 36000, 'skill': 'creatanything', 'other': u'王母娘娘不是他的老婆'}
# godslaver = {'name': u'月老', 'age': 'unknown', 'other': u'他的老婆叫孟婆'}
# post_info.insert(tweets)

# print post_info.find_one({'name':u'极客'})

# for post in post_info.find():
#     print post
# post_info.insert(god)
# post_info.insert(godslaver)
# post_info.remove({'name': u'极客'})
for u in post_info.find().sort([("date", pymongo.ASCENDING)]):
    post_info2.insert(u)
    # print u
print u'操作数据库完成！'
file.close