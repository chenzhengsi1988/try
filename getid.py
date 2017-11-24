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

remote_addr=[]

file=open("/Users/zsc/Desktop/test.txt", 'r')
for line in file :
    tweets.append(json.loads(line))
file.close


for line in tweets:


    # location= line['location']
    remote_addr.append(line['remote_addr'])





newdata={'lremote_addr':remote_addr}
print(newdata)
frame =pd.DataFrame(newdata)

print(frame)

pd.DataFrame.to_csv(frame,'~/ml/data/remote_addrdata.txt')



sortlist=['status','location_speed','lremote_addr','httpcode_5m_200','httpcode_5m_302','httpcode_5m_404','httpcode_5m_403',\
'httpcode_5m_500','httpcode_30m_200','httpcode_30m_302','httpcode_30m_404','httpcode_30m_403','httpcode_30m_500','httpcode_1d_200',\
'httpcode_1d_302','httpcode_1d_404','httpcode_1d_403','httpcode_1d_500','body_bytes_sent','upstream_response_time','httpcode_total_200',\
'httpcode_total_302','httpcode_total_404','httpcode_total_403','httpcode_total_500','request_time','location_city','is_good']

print(sortlist)