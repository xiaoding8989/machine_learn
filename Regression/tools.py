# usr/bin/env python
# -*-coding:utf-8 -*-
"""
liuyading
theme:
"""
import csv
csvfile = file('multi_data1.csv', 'wb')
writer = csv.writer(csvfile)
data=[
    (100,4,9.3),
    (50,3,4.8),
    (100,4,8.9),
    (100,2,6.5),
    (50,2,4.2),
    (80,2,6.2),
    (75,3,7.4),
    (65,4,6),
    (90,3,7.6),
    (90,2,6.1)
]
writer.writerows(data)
csvfile.close()
