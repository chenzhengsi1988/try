# -*- coding: utf-8 -*-
import threading
import time
def fun_timer():
    print('Hello Timer!')
    # global timer
    # timer = threading.Timer(0.5, fun_timer)
    # timer.start()

timer = threading.Timer(1, fun_timer)
timer.start()

time.sleep(10) # 15秒后停止定时器
timer.cancel()