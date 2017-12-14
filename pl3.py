# -*- coding: utf-8 -*-
import json
import pandas as pd
from pandas import Series, DataFrame
from sklearn import preprocessing
import numpy as np
import csv
import sklearn
import random

def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
    csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]


name=['status',
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
undefined=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
           0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
           0.0,0.0,0.0,0.0,0.0,u'IP缺失',u'城市缺失',0]
type=['float','float','float','float','float','float','float','float','float','float','float',
      'float','float','float','float','float','float','float','float','float','float',
      'float', 'float', 'float', 'float', 'float','str','str','int']

with open('/Users/zsc/ml/DNN/tf_dataset_and_estimator_apis/dataset/mean_stdspeed1213.csv', 'rb') as f:
    reader2 = unicode_csv_reader(f)
    meanstd = list(reader2)
# print(meanstd)
meanp=[]
stdp=[]
fs1=len(meanstd)
print(fs1)
for i in range(1, fs1):
    meanp=np.append(meanp,float(meanstd[i][0]))
    stdp = np.append(stdp, float(meanstd[i][1]))


meanp=np.append(meanp, ['/','/','/'])
stdp=np.append(stdp, ['/','/','/'])
print(meanp)
print(stdp)

pl={'name':name,
    'undefined':undefined,
    'type':type,
    'mean': meanp,
    'std': stdp
    }
fpl =pd.DataFrame(pl)
sortlist=['name','undefined','type','mean','std']
fpl=fpl.reindex(columns=sortlist)
pd.DataFrame.to_csv(fpl,'~/ml/data/pl3.csv',encoding='utf8',index=None)
