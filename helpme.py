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

file=open("/Users/zsc/Desktop/test.txt", 'r')
for line in file :
    tweets.append(json.loads(line))
file.close


for line in tweets:


    location= line['location']
    print(location)

    print(location['speed'])
    print(location['city'])
    # location_city.append(location['city'])




# print(frame)