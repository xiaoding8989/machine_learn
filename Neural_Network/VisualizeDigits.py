# usr/bin/env python
# -*-coding:utf-8 -*-
"""
liuyading
theme:下载自带的数据集
"""
from sklearn.datasets import load_digits
digits=load_digits()
print digits.data.shape

import pylab as pl

pl.gray()
pl.matshow(digits.images[1])
pl.show()