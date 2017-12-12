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

meanp=[239.5245,
5544.8045,
28.764,
1.3965,
1.828,
1.614,
1.8415,
169.096,
8.2405,
10.653,
9.412,
10.8025,
4133.606,
197.4445,
254.979,
238.154,
256.8605,
17633,
0.933909491837,
5676.65,
279.5565,
360.243,
323.778,
360.2725,
17633]
stdp=[87.0230739502,
18386.4071925,
4.86315782183,
2.02466978789,
2.39883638458,
2.3704438403,
2.59371890343,
20.780707014,
5.47810731457,
5.3045820759,
5.35576847894,
5.80168025231,
1472.14299331,
65.2476047051,
85.5037985063,
86.3149482071,
88.1814268412,
17481,
0.0584985586264,
471.663984739,
22.9507256476,
29.3841275351,
24.2824775507,
29.4952240837,
17481]
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
pd.DataFrame.to_csv(fpl,'~/ml/data/pl2.csv',encoding='utf8',index=None)
