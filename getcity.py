#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:20:58 2017

@author: zsc
"""

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




tweets = []

location_city=[]

file=open("/Users/zsc/Desktop/testdata.txt", 'r')
for line in file :
    tweets.append(json.loads(line))
file.close


for line in tweets:


    location= line['location']
    location_city.append(location['city'])




cityset=set(location_city)

num=len(cityset)

print(location_city)
print(cityset)
print(num)
citylist=[]

for each in cityset:

    citylist.append(each)
    print(each)

listnum=range(0,num)
print(listnum)



citydict=dict(zip(citylist,listnum))
print(citydict)



embeds = nn.Embedding(num, 5)
city_embed={}
for key in citydict:
    city_idx = torch.LongTensor([citydict[key]])
    city_idx = Variable(city_idx)
    city_embed[key] = embeds(city_idx)
print(city_embed)
print(type(city_embed[key]))
