# usr/bin/env python
# -*-coding:utf-8 -*-
"""
liuyading
theme:
"""
from sklearn import svm

X = [[2, 0], [1, 1], [2,3]]

#三个点归类的标记在超平面下方的也就是[2, 0], [1, 1]为0[2,3]为1
y = [0, 0, 1]
#svm调用svc这个方程  一个线性核函数
clf = svm.SVC(kernel = 'linear')
#调用fit建立模型，调用fit把超平面给算出来了，所用的函数，属性都保存在clf这个分类器里面
clf.fit(X, y)

print clf

# get support vectors
print clf.support_vectors_

# get indices of support vectors找到支持向量点的索引
print clf.support_

# get number of support vectors for each class每个边际有多少个点是支持向量
print clf.n_support_

#进行预测
print clf.predict([2,3])


#2 sklearn画出决定界限


