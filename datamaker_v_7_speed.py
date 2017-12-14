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
import random

tweets = []
status=[]

location_speed=[]
location_distance=[]
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
label=[]

file=open("/Users/zsc/Desktop/result.txt", 'r')
for line in file :
    tweets.append(json.loads(line))
file.close

sortlist=['status',
 'location_speed',
 'location_distance',
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
 'remote_addr',
 'location_city',
 'label']


sortlist2=[
 'status',
 'location_speed',
 'location_distance',
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
 'request_time'
]

sortlist3=[
 'remote_addr',
 'location_city',
 'label']

flag=0
sum11=0
sum21=0
sum31=0
sum41=0

for line in tweets:

    status.append(int(line['status']))

    location = line['location']
    dis=float(location['distance'])
    location_distance.append(dis)
    speednum1 = random.randint(0, 9)
    if speednum1 >= 0 and speednum1 <= 8:
        speed = random.randint(0, 400)
        # location['speed'] = str(speed)
        location_speed.append(float(speed))

    else:
        speed = random.randint(401, 100000)
        # location['speed'] = str(speed)
        # if dis>1000 and speed>800:
        location_speed.append(float(speed))
        if dis > 1000 and speed > 800:
            flag=1
        if dis<=1000 and speed>400:
            flag=2


    if 'city' in location:
        location_city.append(location['city'])
        if location['city']==u'上海':
            sum11=sum11+1
            anum = random.randint(0, 9)
            if anum >= 0 and anum <= 7:
                sum21=sum21+1
                # flag=2
    else:
        location_city.append(u'城市缺失')

    remote_addr.append(line['remote_addr'])
    if  line['remote_addr']==u'124.72.93.18':
        sum31=sum31+1
        anum = random.randint(0, 9)
        if anum >= 0 and anum <= 8:
            # flag=3
            sum41=sum41+1



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
    body_bytes_sent.append(float(line['body_bytes_sent']))
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

    if flag==1:
        label.append(int(1))
    elif flag==2:
        label.append(int(2))
    else:
        label.append(int(0))
    flag=0




# print(status)
# print( httpcode_5m_200)
# print( httpcode_5m_302)
# print( httpcode_5m_404)
# print( httpcode_5m_403)
# print( httpcode_5m_500)
# print( httpcode_30m_200)
# print( httpcode_30m_302)
# print( httpcode_30m_404)
# print( httpcode_30m_403)
# print( httpcode_30m_500)
# print(httpcode_1d_200)
# print(httpcode_1d_302)
# print(httpcode_1d_404)
# print(httpcode_1d_403)
# print(httpcode_1d_500)
# print(uid)
print(body_bytes_sent)
# print(upstream_response_time)
# print(tid)
# print(time_local)
# print(httpcode_total_200)
# print(httpcode_total_302)
# print(httpcode_total_404)
# print(httpcode_total_403)
# print(httpcode_total_500)
print(request_time)
# print(label)
print(type(body_bytes_sent))
print(type(request_time))
newdata={


         'status':status,
         'location_speed':location_speed,
         'location_distance':location_distance,
         'location_city':location_city,
         'remote_addr':remote_addr,
         'httpcode_5m_200': httpcode_5m_200,
         'httpcode_5m_302': httpcode_5m_302,
         'httpcode_5m_404': httpcode_5m_404,
         'httpcode_5m_403': httpcode_5m_403,
         'httpcode_5m_500': httpcode_5m_500,
         'httpcode_30m_200': httpcode_30m_200,
         'httpcode_30m_302': httpcode_30m_302,
         'httpcode_30m_404': httpcode_30m_404,
         'httpcode_30m_403': httpcode_30m_403,
         'httpcode_30m_500': httpcode_30m_500,
         'httpcode_1d_200': httpcode_1d_200,
         'httpcode_1d_302': httpcode_1d_302,
         'httpcode_1d_404': httpcode_1d_404,
         'httpcode_1d_403': httpcode_1d_403,
         'httpcode_1d_500': httpcode_1d_500,
         'body_bytes_sent':body_bytes_sent,
         'upstream_response_time':upstream_response_time,
         'httpcode_total_200':httpcode_total_200,
         'httpcode_total_302':httpcode_total_302,
         'httpcode_total_404':httpcode_total_404,
         'httpcode_total_403':httpcode_total_403,
         'httpcode_total_500':httpcode_total_500,
         'request_time':request_time,
         'label':label
         }

frame =pd.DataFrame(newdata)
frame=frame.reindex(columns=sortlist)

framea=frame.reindex(columns=sortlist2)
frameb=frame.reindex(columns=sortlist3)

frameanp=np.array(framea)
# scaler=preprocessing.MinMaxScaler()
# scaler = preprocessing.StandardScaler()
# F_scaled = frameanp.apply(scaler.fit_transform())


# print(frameanp)
# F_scaled = scaler.fit_transform(frameanp)


# scaler=preprocessing.MinMaxScaler()
# F_scaled = scaler.fit_transform(frameanp)
# for num in range(0,3):
#     aaa=frameanp[:,num]
    # scaler.fit_transform(aaa.reshape(1,-1))
F_scaled = sklearn.preprocessing.scale(frameanp)



S_framea=DataFrame(F_scaled)


alldata=[S_framea,frameb]
sframe = pd.concat(alldata,axis=1)
# pd.DataFrame.to_csv(frameanp ,'~/ml/data/good_bad_data_training33.txt',encoding='utf8')


print(frameanp)
# by=DataFrame(frame, columns=['location_city'])

# sframe = frame.sort(columns=sortlist)
# print(by)

pd.DataFrame.to_csv(sframe,'~/ml/data/gooddata_speed1213.txt',encoding='utf8')



status=[]

location_speed=[]
location_distance=[]
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
label=[]

flag=0
sum1=0
sum2=0
sum3=0
sum4=0
for line in tweets:

    status.append(int(line['status']))

    location = line['location']

    dis = float(location['distance'])
    location_distance.append(dis)
    speednum1 = random.randint(0, 9)
    if speednum1 >= 0 and speednum1 <= 8:
        speed = random.randint(0, 400)
        # location['speed'] = str(speed)
        location_speed.append(float(speed))

    else:
        speed = random.randint(401, 100000)
        # location['speed'] = str(speed)
        # if dis>1000 and speed>800:
        location_speed.append(float(speed))
        if dis > 1000 and speed > 800:
            flag = 1
        if dis <= 1000 and speed > 400:
            flag = 2

    if 'city' in location:
        location_city.append(location['city'])
        if location['city']==u'上海':
            anum = random.randint(0, 9)
            sum1=sum1+1
            if anum >= 0 and anum <= 7:
                # flag=1
                sum2=sum2+1

    else:
        location_city.append(u'城市缺失')

    remote_addr.append(line['remote_addr'])
    if  line['remote_addr']==u'124.72.93.18':
        sum3=sum3+1
        anum = random.randint(0, 9)
        if anum >= 0 and anum <= 8:
            # flag=1
            sum4=sum4+1

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
    body_bytes_sent.append(float(line['body_bytes_sent']))
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
    if flag==1:
        label.append(int(1))
    elif flag==2:
        label.append(int(2))
    else:
        label.append(int(0))

    flag=0

newdata2={
         'status':status,
         'location_speed':location_speed,
         'location_distance':location_distance,
         'location_city':location_city,
         'remote_addr':remote_addr,
         'httpcode_5m_200': httpcode_5m_200,
         'httpcode_5m_302': httpcode_5m_302,
         'httpcode_5m_404': httpcode_5m_404,
         'httpcode_5m_403': httpcode_5m_403,
         'httpcode_5m_500': httpcode_5m_500,
         'httpcode_30m_200': httpcode_30m_200,
         'httpcode_30m_302': httpcode_30m_302,
         'httpcode_30m_404': httpcode_30m_404,
         'httpcode_30m_403': httpcode_30m_403,
         'httpcode_30m_500': httpcode_30m_500,
         'httpcode_1d_200':httpcode_1d_200,
         'httpcode_1d_302':httpcode_1d_302,
         'httpcode_1d_404':httpcode_1d_404,
         'httpcode_1d_403':httpcode_1d_403,
         'httpcode_1d_500':httpcode_1d_500,
         'body_bytes_sent':body_bytes_sent,
         'upstream_response_time':upstream_response_time,
         'httpcode_total_200':httpcode_total_200,
         'httpcode_total_302':httpcode_total_302,
         'httpcode_total_404':httpcode_total_404,
         'httpcode_total_403':httpcode_total_403,
         'httpcode_total_500':httpcode_total_500,
         'request_time':request_time,
         'label':label
         }

frame2 =pd.DataFrame(newdata2)
frame2=frame2.reindex(columns=sortlist)

framea2=frame2.reindex(columns=sortlist2)
frameb2=frame2.reindex(columns=sortlist3)

frameanp2=np.array(framea2)

# scaler = preprocessing.StandardScaler()
# scaler=preprocessing.MinMaxScaler()
# F_scaled2 = scaler.fit_transform(frameanp)
# for num in range(0,3):
#     aaa=frameanp2[:,num]
#     # print(num)
    # print(aaa)
    # scaler.fit_transform(aaa.reshape(1,-1))

    # print(bbb)
meanp=np.mean(frameanp,axis=0)
# print(meanp.shape)
stdp=np.std(frameanp,axis=0)
# print(stdp.shape)
# print(frameanp2.shape)
fs1,fs2=frameanp2.shape
# print(fs1)
# print(fs2)
# print(meanp)
# print(stdp)
# print(frameanp2.shape)

F_scaled2=np.zeros((fs1,fs2))
# print(F_scaled3[1][1])
for i in range(0,fs1):
    for j in range(0,fs2):
        F_scaled2[i][j]=(frameanp2[i][j]-meanp[j])/stdp[j]



# F_scaled3=np.zeros(frameanp2.shape)
# print(F_scaled3[1][1])
m_s={'mean':meanp,
     'std':stdp}
fm_s =pd.DataFrame(m_s)
# print(F_scaled3.shape)
# F_scaled2=sklearn.preprocessing.scale(frameanp2)
pd.DataFrame.to_csv(fm_s,'~/ml/DNN/tf_dataset_and_estimator_apis/dataset/mean_stdspeed1213.csv',encoding='utf8',index=None)
pd.DataFrame.to_csv(fm_s,'~/ml/DNN/tf_dataset_and_estimator_apis/dataset/mean_std_v_speed1213.csv',encoding='utf8',index=None)


    # print(F_scaled2[:,num])



S_framea2=DataFrame(F_scaled2)


alldata=[S_framea2,frameb2]
sframe2 = pd.concat(alldata,axis=1)


pd.DataFrame.to_csv(S_framea2,'~/ml/data/baddata21213.txt',encoding='utf8')
pd.DataFrame.to_csv(sframe2,'~/ml/data/baddata212123444.txt',encoding='utf8')

alldata=[sframe[0:1500],sframe2[0:500]]
alldata5=[frame[0:1500],frame2[0:500]]

result = pd.concat(alldata)

result5 = pd.concat(alldata5)
pd.DataFrame.to_csv(result,'~/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_trainingspeed1213.csv',encoding='utf8',index=None,header=None)

pd.DataFrame.to_csv(result,'~/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_training_v_speed1213.csv',encoding='utf8',index=None,header=None)

pd.DataFrame.to_csv(result5,'~/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_training_speed1213.csv',encoding='utf8',index=None)


alldata2=[sframe[1501:1999],sframe2[501:1000]]
alldata3=[frame[1501:1999],frame2[501:1000]]
result2 = pd.concat(alldata2)
result3 = pd.concat(alldata3)
pd.DataFrame.to_csv(result3,'~/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_testspeed1213.csv',encoding='utf8',index=None,header=None)
pd.DataFrame.to_csv(result2,'~/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_test000vspeed1213.csv',encoding='utf8',index=None,header=None)
pd.DataFrame.to_csv(result3,'~/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_test_v_speed1213.csv',encoding='utf8',index=None)
# print(type(body_bytes_sent))
# print(type(request_time))