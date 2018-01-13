#coding:UTF-8
import time
import numpy as np
import pandas as pd
import math
import json
tweets=[]
flist=[]

file=open("/Users/zsc/Desktop/user-13885241-9d05-461f-822e-28f9c045de0e.txt", 'r')
uid='user-13885241-9d05-461f-822e-28f9c045de0e'
# print(file)
for line in file :
    tweets.append(json.loads(line))
# print(tweets)
# tweets['request']
for line in tweets:
    alldata=line

for key in alldata:
    flist.append(key)
print flist
    # request=line['request']
    # login = line['login']
    # logout = line['logout']
    # SP_UPDATE = line['SP_UPDATE']
    # default = line['default']
    # USER_CREATE = line['USER_CREATE']
    # USER_DELETE = line['USER_DELETE']
    # IDP_UPDATE = line['IDP_UPDATE']
    # SP_CREATE = line['SP_CREATE']
    # SP_DELETE = line['SP_DELETE']
    # UAC_CREATE = line['UAC_CREATE']
    # UAC_DELETE = line['UAC_DELETE']
    # status.append(int(line['status']))

dt1 = "2016-12-01 00:00:00"
dt2="2017-11-01 00:00:00"
timeArray1 = time.strptime(dt1,"%Y-%m-%d %H:%M:%S")
timeArray2 = time.strptime(dt2,"%Y-%m-%d %H:%M:%S")

# a周简写 b月份简写 d日期 HMS 时分秒 Y年份 m月份
#转换成时间戳
timestamp_begin = time.mktime(timeArray1)
timestamp_end = time.mktime(timeArray2)
tlist = []
for itime in range(int(timestamp_begin), int(timestamp_end)+3600, 3600):
    tlist.append(itime)
# print len(tlist)
for fname in  flist:
    # print fname
    plist = alldata[fname]
    # print len(plist)
    tp = {'timestamp': tlist,
          fname: plist
          }
    ftp = pd.DataFrame(tp)
    sortlist = ['timestamp', fname]
    ftp = ftp.reindex(columns=sortlist)
    pd.DataFrame.to_csv(ftp, '/Users/zsc/source/egads/src/test/resources/train_sid3/' + uid+'_'+fname+ '.csv', encoding='utf8', index=None)
    pd.DataFrame.to_csv(ftp, '/Users/zsc/source/egads/src/test/resources/test_sid3/' + uid+'_'+ fname+ '.csv', encoding='utf8', index=None)



