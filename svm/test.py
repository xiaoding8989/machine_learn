# usr/bin/env python
# -*-coding:utf-8 -*-
"""
liuyading
theme:
"""
import csv

fieldnames = ['RID', 'age', 'income', 'student','credit_rating','class:buy_computer']
rows = [
{'RID': '1', 'age': 'youth',       'income': 'high',  'student':'no','credit_rating':   'fair',    'class:buy_computer':'no'},
{'RID': '2', 'age': 'youth',       'income': 'high',  'student': 'no','credit_rating': 'excellent',  'class:buy_computer': 'no'},
{'RID': '3', 'age': 'middle_aged', 'income': 'high',  'student': 'no', 'credit_rating': 'fair',      'class:buy_computer': 'yes'},
{'RID': '4', 'age': 'senior',      'income': 'medium', 'student': 'no', 'credit_rating': 'fair',    'class:buy_computer': 'yes'},
{'RID': '5', 'age': 'senior',      'income': 'low',    'student':'yes','credit_rating':'fair',       'class:buy_computer':'yes'},
{'RID': '6', 'age': 'senior',      'income': 'low',    'student': 'yes','credit_rating': 'excellent','class:buy_computer': 'no'},
{'RID': '7', 'age': 'middle_aged', 'income': 'low',    'student': 'yes', 'credit_rating': 'excellent','class:buy_computer': 'yes'},
{'RID': '8', 'age': 'youth',       'income': 'medium',  'student': 'no', 'credit_rating': 'fair',    'class:buy_computer': 'no'},
{'RID': '9', 'age': 'youth',       'income': 'low',     'student':'yes','credit_rating':'fair',     'class:buy_computer':'yes'},
{'RID': '10', 'age': 'senior',     'income': 'medium',  'student': 'yes','credit_rating': 'fair',    'class:buy_computer': 'yes'},
{'RID': '11', 'age': 'youth',      'income': 'medium',  'student': 'yes', 'credit_rating': 'excellent', 'class:buy_computer': 'yes'},
{'RID': '12', 'age': 'middle_aged', 'income': 'medium',  'student': 'no', 'credit_rating': 'excellent', 'class:buy_computer': 'yes'},
{'RID': '13', 'age': 'middle_aged', 'income': 'high',    'student': 'yes', 'credit_rating': 'fair',    'class:buy_computer': 'yes'},
{'RID': '14', 'age': 'senior',      'income': 'medium',  'student': 'no', 'credit_rating': 'excellent',  'class:buy_computer': 'no'},





        ]
dict_writer = csv.DictWriter(file('your.csv', 'wb'), fieldnames=fieldnames)
dict_writer.writeheader()  # 这里写header
dict_writer.writerows(rows)  # rows就是表单需要写到csv文件的数据