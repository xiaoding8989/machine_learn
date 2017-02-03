# usr/bin/env python
# -*-coding:utf-8 -*-
"""
liuyading
theme:利用神经网络算法进行手写数字识别
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

#因为sklean不接受0到9之间的数字，所以需要转换为0到1之间的数字
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)

print "start fitting"
nn.fit(X_train,labels_train,epochs=3000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i] )
    #从中选择出预测概率最大的
    predictions.append(np.argmax(o))

    #将预测结果绘图出来
print confusion_matrix(y_test,predictions)
print classification_report(y_test,predictions)



"""
start fitting
[[46  0  0  0  0  0  0  0  0  0]  对角线上的都是预测对的情况10个数字每个数字一行
 [ 0 32  1  0  0  0  1  0  6  3]
 [ 0  0 43  0  0  0  0  1  0  0]
 [ 0  0  0 30  0  0  0  1  2  0]
 [ 0  4  0  0 57  0  0  1  0  0]
 [ 0  0  0  0  1 41  0  0  0  0]
 [ 0  1  0  0  0  0 43  0  0  0]
 [ 0  0  0  0  0  0  0 44  0  1]
 [ 0  1  0  0  0  0  0  0 39  0]
 [ 0  0  0  2  0  0  0  0  4 45]]
             precision    recall  f1-score   support

          0       1.00      1.00      1.00        46
          1       0.84      0.74      0.79        43
          2       0.98      0.98      0.98        44
          3       0.94      0.91      0.92        33
          4       0.98      0.92      0.95        62
          5       1.00      0.98      0.99        42
          6       0.98      0.98      0.98        44
          7       0.94      0.98      0.96        45
          8       0.76      0.97      0.86        40
          9       0.92      0.88      0.90        51

avg / total       0.94      0.93      0.93       450

precision所有预测是0的情况果然是0
recall 在所有是0的情况中预测是0的概率

"""
