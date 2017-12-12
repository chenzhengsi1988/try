# -*- coding: utf-8 -*-
import json
import pandas as pd
from pandas import Series, DataFrame
from sklearn import preprocessing
import numpy as np

import sklearn
import random



number=['0','1','2','3']
type=['正常','30分钟404报错异常','城市异常','IP地址异常']





pl={'number':number,
    'type':type
    }
fpl =pd.DataFrame(pl)
sortlist=['number','type']
fpl=fpl.reindex(columns=sortlist)
pd.DataFrame.to_csv(fpl,'~/nl.csv',encoding='utf8',index=None)
