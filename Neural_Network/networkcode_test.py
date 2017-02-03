# usr/bin/env python
# -*-coding:utf-8 -*-
"""
liuyading
theme:
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from network_demo1 import *
from sklearn.cross_validation import train_test_split


digits = load_digits()
#data为特征量
X = digits.data
#target为0到9之间的值
y = digits.target
#把所有的值转到0到1之间
X -= X.min() # normalize the values to bring them into the range 0-1
X /= X.max()

#输入层64总是与特征向量的维度是一样的，10输出层总是与要区分的个数一样的（0到9总共10个数字）
nn = NeuralNetwork([64,100,10],'logistic')
X_train, X_test, y_train, y_test = train_test_split(X, y)

#因为sklean不接受0到9之间的数字，所以需要转换为0到1之间的数字转换的是标签数据集X已经转好了在上面
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)

print "start fitting"
#假设循环3000次
nn.fit(X_train,labels_train,epochs=3000)


predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i] )
    predictions.append(np.argmax(o))
print confusion_matrix(y_test,predictions)

print classification_report(y_test,predictions)
