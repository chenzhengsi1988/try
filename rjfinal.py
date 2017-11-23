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


file=open("/Users/zsc/Desktop/testdata.txt", 'r')
for line in file :
    tweets.append(json.loads(line))

#    print(tweets)
file.close
print(tweets[1])
print(tweets[2])
print(type(tweets[1]))
print(len(tweets[1]))

aaa=tweets[1]
print(aaa['http_authorization'])
print(type(aaa['http_authorization']))
print(aaa['status'])
print(type(aaa['status']))
print(aaa['httpcode_1d'])
print(type(aaa['httpcode_1d']))
print(aaa['http_user_agent'])
print(type(aaa['http_user_agent']))

print(aaa['uid'])
print(type(aaa['uid']))

print(aaa['http_referer'])
print(type(aaa['http_referer']))


print(aaa['remote_addr'])
print(type(aaa['remote_addr']))

print(aaa['http_x_forwarded_for'])
print(type(aaa['http_x_forwarded_for']))

print(aaa['httpcode_5m'])
print(type(aaa['httpcode_5m']))

print(aaa['request'])
print(type(aaa['request']))

print(aaa['body_bytes_sent'])
print(type(aaa['body_bytes_sent']))

print(aaa['httpcode_0.5h'])
print(type(aaa['httpcode_0.5h']))

print(aaa['v3'])
print(type(aaa['v3']))

print(aaa['http_host'])
print(type(aaa['http_host']))


print(aaa['location'])
print(type(aaa['location']))

print(aaa['upstream_response_time'])
print(type(aaa['upstream_response_time']))

print(aaa['tid'])
print(type(aaa['tid']))

print(aaa['time_local'])
print(type(aaa['time_local']))

print(aaa['http_content_type'])
print(type(aaa['http_content_type']))

print(aaa['httpcode_total'])
print(type(aaa['httpcode_total']))

print(aaa['request_time'])
print(type(aaa['request_time']))



#file = open("/Users/zsc/Desktop/testdata.txt",'r')
#for line in file.readlines():
#    dic = json.loads(line)
#    print(dic)