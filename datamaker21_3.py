#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:20:58 2017

@author: zsc
"""

import json
import pandas as pd
from pandas import Series, DataFrame
from sklearn import preprocessing
import numpy as np

import sklearn


tweets = []
status=[]

location_speed=[]
location_city=[]
remote_addr=[]

httpcode_5m_200=[]
httpcode_5m_302=[]
httpcode_5m_404=[]
httpcode_5m_403=[]
httpcode_5m_500=[]
httpcode_30m_200=[]
httpcode_30m_302=[]
httpcode_30m_404=[]
httpcode_30m_403=[]
httpcode_30m_500=[]
httpcode_1d_200=[]
httpcode_1d_302=[]
httpcode_1d_404=[]
httpcode_1d_403=[]
httpcode_1d_500=[]
uid=[]
body_bytes_sent=[]
upstream_response_time=[]
tid=[]
time_local=[]
httpcode_total_200=[]
httpcode_total_302=[]
httpcode_total_404=[]
httpcode_total_403=[]
httpcode_total_500=[]
request_time=[]
is_good=[]

file=open("/Users/zsc/Desktop/test.txt", 'r')
for line in file :
    tweets.append(json.loads(line))
file.close

sortlist=['status',
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
 'request_time',
 'lremote_addr',
 'location_city',
 'is_good']


sortlist2=[
 'location_speed',
 'httpcode_30m_404',
 'body_bytes_sent',
]

sortlist3=[
 'is_good']



for line in tweets:

    status.append(int(line['status']))

    location = line['location']
    location_speed.append(float(location['speed']))

    if 'city' in location:
        location_city.append(location['city'])
    else:
        location_city.append(u'城市缺失')

    remote_addr.append(line['remote_addr'])

    line_httpcode_5m = line['httpcode_5m']

    if '200' in line_httpcode_5m:
        httpcode_5m_200.append(int(line_httpcode_5m['200']))
    else:
        httpcode_5m_200.append(int(0))

    if '302' in line_httpcode_5m:
        httpcode_5m_302.append(int(line_httpcode_5m['302']))
    else:
        httpcode_5m_302.append(int(0))

    if '404' in line_httpcode_5m:
        httpcode_5m_404.append(int(line_httpcode_5m['404']))
    else:
        httpcode_5m_404.append(int(0))

    if '403' in line_httpcode_5m:
        httpcode_5m_403.append(int(line_httpcode_5m['403']))
    else:
        httpcode_5m_403.append(int(0))

    if '500' in line_httpcode_5m:
        httpcode_5m_500.append(int(line_httpcode_5m['500']))
    else:
        httpcode_5m_500.append(int(0))

    line_httpcode_30m = line['httpcode_0.5h']

    if '200' in line_httpcode_30m:
        httpcode_30m_200.append(int(line_httpcode_30m['200']))
    else:
        httpcode_30m_200.append(int(0))

    if '302' in line_httpcode_30m:
        httpcode_30m_302.append(int(line_httpcode_30m['302']))
    else:
        httpcode_30m_302.append(int(0))

    if '404' in line_httpcode_30m:
        httpcode_30m_404.append(int(line_httpcode_30m['404']))
    else:
        httpcode_30m_404.append(int(0))

    if '403' in line_httpcode_30m:
        httpcode_30m_403.append(int(line_httpcode_30m['403']))
    else:
        httpcode_30m_403.append(int(0))

    if '500' in line_httpcode_30m:
        httpcode_30m_500.append(int(line_httpcode_30m['500']))
    else:
        httpcode_30m_500.append(int(0))

    line_httpcode_1d=line['httpcode_1d']
    httpcode_1d_200.append(int(line_httpcode_1d['200']))
    httpcode_1d_302.append(int(line_httpcode_1d['302']))
    httpcode_1d_404.append(int(line_httpcode_1d['404']))
    httpcode_1d_403.append(int(line_httpcode_1d['403']))
    httpcode_1d_500.append(int(line_httpcode_1d['500']))
    uid.append(int(line['uid']))
    body_bytes_sent.append(int(line['body_bytes_sent']))
    upstream_response_time.append(float(line['upstream_response_time']))
    tid.append(int(line['tid']))
    time_local.append(int(line['time_local']))
    line_httpcode_total = line['httpcode_total']
    httpcode_total_200.append(int(line_httpcode_total['200']))
    httpcode_total_302.append(int(line_httpcode_total['302']))
    httpcode_total_404.append(int(line_httpcode_total['404']))
    httpcode_total_403.append(int(line_httpcode_total['403']))
    httpcode_total_500.append(int(line_httpcode_total['500']))
    request_time.append(float(line['request_time']))
    is_good.append(int(1))




print(status)
print( httpcode_5m_200)
print( httpcode_5m_302)
print( httpcode_5m_404)
print( httpcode_5m_403)
print( httpcode_5m_500)
print( httpcode_30m_200)
print( httpcode_30m_302)
print( httpcode_30m_404)
print( httpcode_30m_403)
print( httpcode_30m_500)
print(httpcode_1d_200)
print(httpcode_1d_302)
print(httpcode_1d_404)
print(httpcode_1d_403)
print(httpcode_1d_500)
print(uid)
print(body_bytes_sent)
print(upstream_response_time)
print(tid)
print(time_local)
print(httpcode_total_200)
print(httpcode_total_302)
print(httpcode_total_404)
print(httpcode_total_403)
print(httpcode_total_500)
print(request_time)
print(is_good)
newdata={


         # 'status':status,

         'location_speed':location_speed,
         # 'location_city':location_city,
         # 'lremote_addr':remote_addr,
         # 'httpcode_5m_200': httpcode_5m_200,
         # 'httpcode_5m_302': httpcode_5m_302,
         # 'httpcode_5m_404': httpcode_5m_404,
         # 'httpcode_5m_403': httpcode_5m_403,
         # 'httpcode_5m_500': httpcode_5m_500,
         # 'httpcode_30m_200': httpcode_30m_200,
         # 'httpcode_30m_302': httpcode_30m_302,
         'httpcode_30m_404': httpcode_30m_404,
         # 'httpcode_30m_403': httpcode_30m_403,
         # 'httpcode_30m_500': httpcode_30m_500,
         # 'httpcode_1d_200': httpcode_1d_200,
         # 'httpcode_1d_302': httpcode_1d_302,
         # 'httpcode_1d_404': httpcode_1d_404,
         # 'httpcode_1d_403': httpcode_1d_403,
         # 'httpcode_1d_500': httpcode_1d_500,
         'body_bytes_sent':body_bytes_sent,
         # 'upstream_response_time':upstream_response_time,
         # 'httpcode_total_200':httpcode_total_200,
         # 'httpcode_total_302':httpcode_total_302,
         # 'httpcode_total_404':httpcode_total_404,
         # 'httpcode_total_403':httpcode_total_403,
         # 'httpcode_total_500':httpcode_total_500,
         # 'request_time':request_time,
         'is_good':is_good
         }

frame =pd.DataFrame(newdata)
frame=frame.reindex(columns=sortlist)

framea=frame.reindex(columns=sortlist2)
frameb=frame.reindex(columns=sortlist3)

frameanp=np.array(framea)
scaler=preprocessing.MinMaxScaler()
# scaler = preprocessing.StandardScaler()
# F_scaled = frameanp.apply(scaler.fit_transform())



# F_scaled = scaler.fit_transform(frameanp)


scaler=preprocessing.MinMaxScaler()
F_scaled = scaler.fit_transform(frameanp)
for num in range(0,3):
    aaa=frameanp[:,num]
    # scaler.fit_transform(aaa.reshape(1,-1))
    F_scaled[:, num] = sklearn.preprocessing.normalize(aaa.reshape(1, -1),norm='l2')



S_framea=DataFrame(F_scaled)


alldata=[S_framea,frameb]
sframe = pd.concat(alldata,axis=1)
# pd.DataFrame.to_csv(frameanp ,'~/ml/data/good_bad_data_training33.txt',encoding='utf8')


print(frameanp)
# by=DataFrame(frame, columns=['location_city'])

# sframe = frame.sort(columns=sortlist)
# print(by)

pd.DataFrame.to_csv(sframe,'~/ml/data/gooddata.txt',encoding='utf8')



status=[]

location_speed=[]
location_city=[]
remote_addr=[]

httpcode_5m_200=[]
httpcode_5m_302=[]
httpcode_5m_404=[]
httpcode_5m_403=[]
httpcode_5m_500=[]
httpcode_30m_200=[]
httpcode_30m_302=[]
httpcode_30m_404=[]
httpcode_30m_403=[]
httpcode_30m_500=[]
httpcode_1d_200=[]
httpcode_1d_302=[]
httpcode_1d_404=[]
httpcode_1d_403=[]
httpcode_1d_500=[]
uid=[]
body_bytes_sent=[]
upstream_response_time=[]
tid=[]
time_local=[]
httpcode_total_200=[]
httpcode_total_302=[]
httpcode_total_404=[]
httpcode_total_403=[]
httpcode_total_500=[]
request_time=[]
is_good=[]
for line in tweets[0:3000]:

    status.append(int(line['status']))

    location = line['location']
    location_speed.append(float(location['speed']))

    if 'city' in location:
        location_city.append(location['city'])
    else:
        location_city.append(u'城市缺失')

    remote_addr.append(line['remote_addr'])

    line_httpcode_5m = line['httpcode_5m']

    if '200' in line_httpcode_5m:
        httpcode_5m_200.append(int(line_httpcode_5m['200']))
    else:
        httpcode_5m_200.append(int(0))

    if '302' in line_httpcode_5m:
        httpcode_5m_302.append(int(line_httpcode_5m['302']))
    else:
        httpcode_5m_302.append(int(0))

    if '404' in line_httpcode_5m:
        httpcode_5m_404.append(int(line_httpcode_5m['404']))
    else:
        httpcode_5m_404.append(int(0))

    if '403' in line_httpcode_5m:
        httpcode_5m_403.append(int(line_httpcode_5m['403']))
    else:
        httpcode_5m_403.append(int(0))

    if '500' in line_httpcode_5m:
        httpcode_5m_500.append(int(line_httpcode_5m['500']))
    else:
        httpcode_5m_500.append(int(0))

    line_httpcode_30m = line['httpcode_0.5h']

    if '200' in line_httpcode_30m:
        httpcode_30m_200.append(int(line_httpcode_30m['200']))
    else:
        httpcode_30m_200.append(int(0))

    if '302' in line_httpcode_30m:
        httpcode_30m_302.append(int(line_httpcode_30m['302']))
    else:
        httpcode_30m_302.append(int(0))

    if '404' in line_httpcode_30m:
        httpcode_30m_404.append(int(line_httpcode_30m['404'])+10)
    else:
        httpcode_30m_404.append(int(0)+10)

    if '403' in line_httpcode_30m:
        httpcode_30m_403.append(int(line_httpcode_30m['403']))
    else:
        httpcode_30m_403.append(int(0))

    if '500' in line_httpcode_30m:
        httpcode_30m_500.append(int(line_httpcode_30m['500']))
    else:
        httpcode_30m_500.append(int(0))

    line_httpcode_1d=line['httpcode_1d']
    httpcode_1d_200.append(int(line_httpcode_1d['200']))
    httpcode_1d_302.append(int(line_httpcode_1d['302']))
    httpcode_1d_404.append(int(line_httpcode_1d['404']))
    httpcode_1d_403.append(int(line_httpcode_1d['403']))
    httpcode_1d_500.append(int(line_httpcode_1d['500']))
    uid.append(int(line['uid']))
    body_bytes_sent.append(int(line['body_bytes_sent']))
    upstream_response_time.append(float(line['upstream_response_time']))
    tid.append(int(line['tid']))
    time_local.append(int(line['time_local']))
    line_httpcode_total = line['httpcode_total']
    httpcode_total_200.append(int(line_httpcode_total['200']))
    httpcode_total_302.append(int(line_httpcode_total['302']))
    httpcode_total_404.append(int(line_httpcode_total['404']))
    httpcode_total_403.append(int(line_httpcode_total['403']))
    httpcode_total_500.append(int(line_httpcode_total['500']))
    request_time.append(float(line['request_time']))
    is_good.append(int(0))


newdata2={
    # 'status':status,
         'location_speed':location_speed,
#          'location_city':location_city,
#          'lremote_addr':remote_addr,
#          'httpcode_5m_200': httpcode_5m_200,
#          'httpcode_5m_302': httpcode_5m_302,
#          'httpcode_5m_404': httpcode_5m_404,
#          'httpcode_5m_403': httpcode_5m_403,
#          'httpcode_5m_500': httpcode_5m_500,
#          'httpcode_30m_200': httpcode_30m_200,
#          'httpcode_30m_302': httpcode_30m_302,
         'httpcode_30m_404': httpcode_30m_404,
         # 'httpcode_30m_403': httpcode_30m_403,
         # 'httpcode_30m_500': httpcode_30m_500,
         # 'httpcode_1d_200':httpcode_1d_200,
         # 'httpcode_1d_302':httpcode_1d_302,
         # 'httpcode_1d_404':httpcode_1d_404,
         # 'httpcode_1d_403':httpcode_1d_403,
         # 'httpcode_1d_500':httpcode_1d_500,
         'body_bytes_sent':body_bytes_sent,
         # 'upstream_response_time':upstream_response_time,
         # 'httpcode_total_200':httpcode_total_200,
         # 'httpcode_total_302':httpcode_total_302,
         # 'httpcode_total_404':httpcode_total_404,
         # 'httpcode_total_403':httpcode_total_403,
         # 'httpcode_total_500':httpcode_total_500,
         # 'request_time':request_time,
         'is_good':is_good
         }

frame2 =pd.DataFrame(newdata2)
frame2=frame2.reindex(columns=sortlist)

framea2=frame2.reindex(columns=sortlist2)
frameb2=frame2.reindex(columns=sortlist3)

frameanp2=np.array(framea2)

# scaler = preprocessing.StandardScaler()
scaler=preprocessing.MinMaxScaler()
F_scaled2 = scaler.fit_transform(frameanp)
for num in range(0,3):
    aaa=frameanp2[:,num]
    # print(num)
    # print(aaa)
    # scaler.fit_transform(aaa.reshape(1,-1))

    # print(bbb)
    F_scaled2[:,num]=sklearn.preprocessing.normalize(aaa.reshape(1,-1),norm='l2')
    # print(F_scaled2[:,num])



S_framea2=DataFrame(F_scaled2)


alldata=[S_framea2,frameb2]
sframe2 = pd.concat(alldata,axis=1)


pd.DataFrame.to_csv(sframe2,'~/ml/data/baddata.txt',encoding='utf8')

alldata=[sframe[0:2700],sframe2[0:2700]]
result = pd.concat(alldata)
pd.DataFrame.to_csv(result,'~/ml/data/good_bad_data_training3_3.txt',encoding='utf8',index=None)


alldata2=[sframe[2700:3000],sframe2[2700:3000]]
result2 = pd.concat(alldata2)
pd.DataFrame.to_csv(result2,'~/ml/data/good_bad_data_testing3_3.txt',encoding='utf8',index=None)
