#coding:UTF-8
import time
import numpy as np
import pandas as pd
import math
import random
from pandas import Series, DataFrame
c=math.pi*2/24/7
lam=5
# size=24*28
# np.random.seed(0)
# plist=[]
# tlist=[]
# plist=np.random.poisson(lam, 1)

# dt1 = "2017-11-01 08:00:00"
# dt2="2017-11-29 08:00:00"
# dt1="2014-07-26 12:00:00"
# dt2="2014-08-23 12:00:00"
dt1="2015-06-28 13:00:00"
dt2="2015-07-26 13:00:00"
timeArray1 = time.strptime(dt1,"%Y-%m-%d %H:%M:%S")
timeArray2 = time.strptime(dt2,"%Y-%m-%d %H:%M:%S")

# a周简写 b月份简写 d日期 HMS 时分秒 Y年份 m月份
#转换成时间戳
timestamp_begin = time.mktime(timeArray1)
timestamp_end = time.mktime(timeArray2)
print timestamp_begin
for iuid in range(100,111):
    # np.random.seed(0)
    plist = []
    tlist = []
    ptlist=[]
    tp={}
    tp1={}
    i=0
    # np.random.seed(0)
    # np.random.seed(0)
    A=np.random.randint(1, 9)
    # print A
    for itime in range(int(timestamp_begin), int(timestamp_end), 3600):
        # np.random.seed(0)
        B=np.random.normal(0,1,1)
        # np.random.seed(1)
        # B1=np.random.normal(0,1,1)

        s=10*A*math.sin(c*i)
        s1=10*A*math.sin(c*i)
        if i==2:
            s1=s1+B[0]
        if i == 7:
            s1=s1+B[0]
        # print s
        # print s1
        # print A
        i=i+1
        plist.append(s)
        tlist.append(itime)
        ptlist.append(s1)
        # print plist[0]
        # print ptlist[0]
    tp = {'timestamp': tlist,
          'value': plist
          }
    ftp = pd.DataFrame(tp)
    sortlist = ['timestamp', 'value']
    ftp = ftp.reindex(columns=sortlist)
    pd.DataFrame.to_csv(ftp,'/Users/zsc/ml/data_olympic/train1224/'+ str(iuid)+'.csv',encoding='utf8', index=None)

    tp1 = {'timestamp': tlist,
          'value': ptlist
          }
    ftp1 = pd.DataFrame(tp1)
    sortlist1 = ['timestamp', 'value']
    ftp1 = ftp1.reindex(columns=sortlist1)
    # print ftp
    # print ftp1
    # print ['~/ml/olympic_data/training/'str(iuid),'.csv']
    # pd.DataFrame.to_csv(ftp,'/Users/zsc/ml/data_olympic/training/'+ str(iuid)+'.csv',encoding='utf8', index=None)
    pd.DataFrame.to_csv(ftp1,'/Users/zsc/ml/data_olympic/test1224/'+ str(iuid)+'.csv',encoding='utf8', index=None)


# print ftp
# print ftp1
# for iuid in range(100,201):
#     np.random.seed(0)
#     plist = []
#     tlist = []
#     for itime in range(int(timestamp_begin), int(timestamp_end), 3600):
#         plist.append(np.random.poisson(lam, 1)[0])
#         tlist.append(itime)
#     tp = {'timestamp': tlist,
#           'value': plist
#           }
#     ftp = pd.DataFrame(tp)
#     sortlist = ['timestamp', 'value']
#     ftp = ftp.reindex(columns=sortlist)
#     # print ['~/ml/olympic_data/training/'str(iuid),'.csv']
#     pd.DataFrame.to_csv(ftp,'/Users/zsc/ml/data_olympic/test/'+ str(iuid)+'.csv',encoding='utf8', index=None)


    # np.random.seed(0)
# plist=[]
# tlist=[]
# for itime in range(int(timestamp_begin),int(timestamp_end),3600):
#     plist.append(np.random.poisson(lam, 1)[0])
#     tlist.append(itime)
# tp={'timestamp':tlist,
#     'value':plist
#     }
# ftp=pd.DataFrame(tp)
# sortlist=['timestamp','value']
# ftp=ftp.reindex(columns=sortlist)
# print ftp


# sampleNo = 1000;
# # 一维正态分布
# # 下面三种方式是等效的
# mu = 3
# sigma = 0.1
# np.random.seed(0)
# s = np.random.normal(mu, sigma, sampleNo )
# print s