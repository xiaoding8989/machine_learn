# usr/bin/env python
# -*-coding:utf-8 -*-
"""
liuyading
theme:
X:                  Y
0 0                 0
0 1                 1
1 0                 1
1 1                 0
"""
from network_demo1 import *
import numpy as np

nn = NeuralNetwork([2,2,1], 'tanh')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
nn.fit(X, y)
for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
    print(i, nn.predict(i))