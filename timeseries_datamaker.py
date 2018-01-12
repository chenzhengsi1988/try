#coding:UTF-8
import time
import pandas as pd
import json
import csv
tweets = []
flist = ['request']


file=open("/Users/zsc/Desktop/vegachen@yufuid.com.txt", 'r')

for line in file :
    tweets.append(json.loads(line))

for line in tweets:
    alldata = line


dt1 = "2016-12-01 00:00:00"
dt2 = "2017-11-01 00:00:00"
timeArray1 = time.strptime(dt1, "%Y-%m-%d %H:%M:%S")
timeArray2 = time.strptime(dt2, "%Y-%m-%d %H:%M:%S")

timestamp_begin = time.mktime(timeArray1)
timestamp_end = time.mktime(timeArray2)
tlist = []
for itime in range(int(timestamp_begin), int(timestamp_end)+3600, 3600):
    tlist.append(itime)

plist = alldata['request']
tp = {'timestamp': tlist,
      'request': plist
        }
fieldnames1=['timestamp','request']
listlen =len(plist)

csvFile1 = open('/Users/zsc/source/egads_allseeing/src/test/resources/testing2/request.csv','w' )
writer1 = csv.DictWriter(csvFile1,fieldnames=fieldnames1)
writer1.writeheader()
for i in range(0, listlen):
    writer1.writerow({'timestamp': tlist[i],'request': plist[i]})
csvFile1.close()

csvFile1 = open('/Users/zsc/source/egads_allseeing/src/test/resources/training2/request.csv','w' )
writer1 = csv.DictWriter(csvFile1,fieldnames=fieldnames1)
writer1.writeheader()
for i in range(0, listlen):
    writer1.writerow({'timestamp': tlist[i],'request': plist[i]})
csvFile1.close()
