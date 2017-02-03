# usr/bin/env python
# -*-coding:utf-8 -*-
"""
liuyading
theme:
"""
import numpy as np
import random
def genData(numPoints, bias, variance):
    #行数与列数
    x = np.zeros(shape=(numPoints, 2))
    #y是归类的标签 一列
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        #对行数进行循环第一列与第二列
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        #对y进行赋值
        y[i] = (i + bias) +random.uniform(0,1) * variance
    return x, y

#gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = genData(100, 25, 10)

print x.shape
xTrans = x.transpose()
print xTrans.shape