#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:20:58 2017

@author: zsc
"""

import json
import pandas as pd
import numpy as np

#with open("/Users/zsc/Desktop/testdata.txt") as f:
   # json_data = json.load(f)



tweets = []
status=[]
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


file=open("/Users/zsc/Desktop/testdata.txt", 'r')
for line in file :
    tweets.append(json.loads(line))

# print(tweets)
file.close


for line in tweets:
# print(line)
    status.append(int(line['status']))
    line_httpcode_5m=line['httpcode_5m']

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


    line_httpcode_30m=line['httpcode_0.5h']

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

print(line)
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