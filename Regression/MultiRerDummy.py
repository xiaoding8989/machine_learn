# usr/bin/env python
# -*-coding:utf-8 -*-
"""
liuyading
theme:如果自变量中有分类型变量(categorical data) , 如何处理？
"""
from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model
#linear_model----建模回归分析
#datasets数据集

dataPath = "multi_data.csv"
# genfromtxt方法从指定路径里提取数据转化为numpy格式也就是矩阵格式
deliveryData = genfromtxt(dataPath, delimiter=',')

print "data"
print deliveryData

#将数据拆成2部分
X = deliveryData[:, :-1]
Y = deliveryData[:, -1]

print "X:"
print X
print "Y: "
print Y

#调用线性回归模型
regr = linear_model.LinearRegression()

#调用fit方法对x与Y进行建模
regr.fit(X, Y)

print "coefficients"
#打印b1,b2,b3,b4的值
print regr.coef_
print "intercept: "
#打印b0
print regr.intercept_

xPred = [102,6,0,1,0]
yPred = regr.predict(xPred)
print "predicted y: "
print yPred


