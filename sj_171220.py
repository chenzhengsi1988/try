#coding:UTF-8
import time
import datetime

# dt = "1970-01-02 08:00:00"
dt="Thu Jan 01 08:00:00 CST 1970"

#转换成时间数组
timeArray = time.strptime(dt,"%a %b %d %H:%M:%S CST %Y")
# a周简写 b月份简写 d日期 HMS 时分秒 Y年份 m月份
#转换成时间戳
timestamp = time.mktime(timeArray)
# tt=time.ctime()
# timeArray2 = time.strptime(tt,"%Y-%m-%d %H:%M:%S")

print timestamp

# dt = "2016-05-05 20:28:54"
#
# #转换成时间数组
# timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
# #转换成新的时间格式(20160505-20:28:54)
# dt_new = time.strftime("%Y%m%d-%H:%M:%S",timeArray)
#
# print dt_new

timeStamp = 1403902800
dateArray = datetime.datetime.utcfromtimestamp(timeStamp)
otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
# otherStyletime == "2013-10-10 23:40:00"
print(otherStyleTime)