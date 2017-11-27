#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:20:58 2017

@author: zsc
"""

import json
import pandas as pd
from pandas import Series, DataFrame



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
newdata={'status':status,

         'location_speed':location_speed,
         'location_city':location_city,
         'lremote_addr':remote_addr,
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
         'is_good':is_good
         }

print(newdata)
frame =pd.DataFrame(newdata)
frame=frame.reindex(columns=sortlist)
# by=DataFrame(frame, columns=['location_city'])

# sframe = frame.sort(columns=sortlist)
# print(by)

pd.DataFrame.to_csv(frame,'~/ml/data/gooddata.txt',encoding='utf8')



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
for line in tweets[0:300]:

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


newdata2={'status':status,

         'location_speed':location_speed,
         'location_city':location_city,
         'lremote_addr':remote_addr,
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
         'is_good':is_good
         }

print(newdata2)
frame2 =pd.DataFrame(newdata2)
frame2=frame2.reindex(columns=sortlist)
# sframe2 = frame2.sort(columns=sortlist)
print(frame)

pd.DataFrame.to_csv(frame2,'~/ml/data/baddata.txt',encoding='utf8')


alldata=[frame[0:2700],frame2[0:270]]
result = pd.concat(alldata)
pd.DataFrame.to_csv(result,'~/ml/data/good_bad_data_training.txt',encoding='utf8',index_label=sortlist,index=None)


alldata2=[frame[2700:3000],frame2[270:300]]
result2 = pd.concat(alldata2)
pd.DataFrame.to_csv(result2,'~/ml/data/good_bad_data_testing.txt',encoding='utf8',index_label=sortlist,index=None)