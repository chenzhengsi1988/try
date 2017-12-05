# -*- coding: utf-8 -*-
import json
import pandas as pd
from pandas import Series, DataFrame
from sklearn import preprocessing
import numpy as np

import sklearn
import random



name=['status',
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
 'remote_addr',
 'location_city',
 'label']
undefined=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
           0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
           0.0,0.0,0.0,0.0,0.0,u'IP缺失',u'城市缺失',0]
type=['float','float','float','float','float','float','float','float','float','float',
      'float','float','float','float','float','float','float','float','float','float',
      'float', 'float', 'float', 'float', 'float','str','str','int']


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
label=[]

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
 'remote_addr',
 'location_city',
 'label']


sortlist2=[
 'status',
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
    location_speed.append(float(location['speed']))

    if 'city' in location:
        location_city.append(location['city'])
        if location['city']==u'上海':
            sum11=sum11+1
            anum = random.randint(0, 9)
            if anum >= 0 and anum <= 7:
                sum21=sum21+1
                flag=2
    else:
        location_city.append(u'城市缺失')

    remote_addr.append(line['remote_addr'])
    if  line['remote_addr']==u'124.72.93.18':
        sum31=sum31+1
        anum = random.randint(0, 9)
        if anum >= 0 and anum <= 8:
            flag=3
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

    if flag==2:
        label.append(int(2))
    elif flag==3:
        label.append(int(3))
    else:
        label.append(int(0))
    flag=0


newdata={


         'status':status,
         'location_speed':location_speed,
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

meanp=np.mean(frameanp,axis=0)
# print(meanp.shape)
stdp=np.std(frameanp,axis=0)
# print(stdp.shape)
# print(frameanp2.shape)
# fs1,fs2=frameanp2.shape
# print(fs1)
# print(fs2)
# print(meanp)
# print(stdp)
# print(frameanp2.shape)

# F_scaled2=np.zeros((fs1,fs2))
# # print(F_scaled3[1][1])
# for i in range(0,fs1):
#     for j in range(0,fs2):
#         F_scaled2[i][j]=(frameanp2[i][j]-meanp[j])/stdp[j]

meanp=np.append(meanp, ['/','/','/'])
stdp=np.append(stdp, ['/','/','/'])
print(meanp)

pl={'name':name,
    'undefined':undefined,
    'type':type,
    'mean': meanp,
    'std': stdp
    }
fpl =pd.DataFrame(pl)
sortlist=['name','undefined','type','mean','std']
fpl=fpl.reindex(columns=sortlist)
pd.DataFrame.to_csv(fpl,'~/ml/data/pl.csv',encoding='utf8',index=None)
