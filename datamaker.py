#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:20:58 2017

@author: zsc
"""

import json
import pandas as pd




tweets = []
status=[]
httpcode_200=[]
httpcode_302=[]
httpcode_404=[]
httpcode_403=[]
httpcode_500=[]
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

file=open("/Users/zsc/Desktop/testdata.txt", 'r')
for line in file :
    tweets.append(json.loads(line))
file.close


for line in tweets:

    status.append(int(line['status']))
    line_httpcode=line['httpcode_1d']
    httpcode_200.append(int(line_httpcode['200']))
    httpcode_302.append(int(line_httpcode['302']))
    httpcode_404.append(int(line_httpcode['404']))
    httpcode_403.append(int(line_httpcode['403']))
    httpcode_500.append(int(line_httpcode['500']))
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
print(httpcode_200)
print(httpcode_302)
print(httpcode_404)
print(httpcode_403)
print(httpcode_500)
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
         'httpcode_200':httpcode_200,
         'httpcode_302':httpcode_302,
         'httpcode_404':httpcode_404,
         'httpcode_403':httpcode_403,
         'httpcode_500':httpcode_500,
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
print(frame)

pd.DataFrame.to_csv(frame,'~/ml/data/gooddata.txt')



status=[]
httpcode_200=[]
httpcode_302=[]
httpcode_404=[]
httpcode_403=[]
httpcode_500=[]
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
for line in tweets[0:1999]:

    status.append(int(line['status']))
    line_httpcode=line['httpcode_1d']
    httpcode_200.append(int(line_httpcode['200']))
    httpcode_302.append(int(line_httpcode['302']))
    httpcode_404.append(int(line_httpcode['404']))
    httpcode_403.append(int(line_httpcode['403']))
    httpcode_500.append(int(line_httpcode['500']))
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
    request_time.append(float(line['request_time'])+1)
    is_good.append(int(0))


newdata2={'status':status,
         'httpcode_200':httpcode_200,
         'httpcode_302':httpcode_302,
         'httpcode_404':httpcode_404,
         'httpcode_403':httpcode_403,
         'httpcode_500':httpcode_500,
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
print(frame2)

pd.DataFrame.to_csv(frame2,'~/ml/data/baddata.txt')


alldata=[frame,frame2]
result = pd.concat(alldata)
pd.DataFrame.to_csv(result,'~/ml/data/good_bad_data.txt')