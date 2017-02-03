# usr/bin/env python
# -*-coding:utf-8 -*-
"""
liuyading
theme:非线性回归的应用
梯度下降算法的代码实现（很重要哦）
"""
import numpy as np
import random

# m denotes the number of examples here, not the number of features
#theta是要学习的参数值Θ（θ0，θ1，θ2，···θn）
#alpha学习率梯度下降算法中下降了多少
#m是多少个实例
#numIterations对更新法则重复了多少次
def gradientDescent(x, y, theta, alpha, m, numIterations):
    #进行矩阵转置
    xTrans = x.transpose()
    for i in range(0, numIterations):
        #相当于h function
        hypothesis = np.dot(x, theta)
        #预测值减去实际值
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        #除以2乘以实例的多少也就是M
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        #gradient为每次更新的多少
        #更新法则
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta

#numPoints多少行多少个实例 bias偏向值variance方差离散度的衡量
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

# print "x:"
# print x
# print "y:"
# print y

m, n = np.shape(x)
numIterations= 100000
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)