#coding:UTF-8
import time
import numpy as np
lam=5
size=24*28
np.random.seed(0)
plist=[]
plist=np.random.poisson(lam, size)
print plist


# sampleNo = 1000;
# # 一维正态分布
# # 下面三种方式是等效的
# mu = 3
# sigma = 0.1
# np.random.seed(0)
# s = np.random.normal(mu, sigma, sampleNo )
# print s