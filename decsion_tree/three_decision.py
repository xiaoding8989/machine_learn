# usr/bin/env python
# -*-coding:utf-8 -*-
"""
liuyading
theme:
"""
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO
allElectronicsData=open(r'your.csv')
reader=csv.reader(allElectronicsData)
headers=reader.next()
print headers

featureList=[]
labelList=[]
for row in reader:
    labelList.append(row[len(row)-1])
    rowDict={}
    for i in range(1,len(row)-1):
        rowDict[headers[i]]=row[i]
    featureList.append(rowDict)

#将dict转换成0与1形式的
vec=DictVectorizer()
dummyX=vec.fit_transform(featureList).toarray()

print("dummyX:"+str(dummyX))
print(vec.get_feature_names())

print("labelList:"+str(labelList))


lb=preprocessing.LabelBinarizer()

dummyY=lb.fit_transform(labelList)

print('dummyY:'+str(dummyY))


#分类器                      决定使用哪个标准在这里使用的事信息熵
clf=tree.DecisionTreeClassifier(criterion='entropy')
#开始建模构建决策树
clf=clf.fit(dummyX,dummyY)

print('clf:'+str(clf))

with open("allElectinfo.dot",'w') as f:
    f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)


oneRowX=dummyX[0,:]
print("oneRowX:"+str(oneRowX))

newRowx=oneRowX
newRowx[0]=1
newRowx[2]=0
print("newRowx:"+str(newRowx))

predictedY=clf.predict(newRowx)
print("predictedY:"+str(predictedY))