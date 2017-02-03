# usr/bin/env python
# -*-coding:utf-8 -*-
"""
liuyading
theme:knn算法，调用python里的内置库实现
"""
from sklearn import neighbors
from sklearn import datasets

#调用分类器
knn = neighbors.KNeighborsClassifier()

#将数据库赋值
iris = datasets.load_iris()

#建立模型
knn.fit(iris.data, iris.target)
#传入实例进行预测
predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])

print predictedLabel
#
# # 3.
# # KNN
# # 实现Implementation:
# # Example of kNN implemented from Scratch in Python
#
