#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:20:58 2017

@author: zsc
"""

import json

#with open("/Users/zsc/Desktop/testdata.txt") as f:
   # json_data = json.load(f)



tweets = []
for line in open("/Users/zsc/Desktop/testdata.txt", 'r'):
    tweets.append(json.loads(line))
print(tweets)




file = open("/Users/zsc/Desktop/testdata.txt",'r')
for line in file.readlines():
    dic = json.loads(line)
    print(dic)