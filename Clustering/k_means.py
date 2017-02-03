# usr/bin/env python
# -*-coding:utf-8 -*-
"""
liuyading
theme:聚类算法k_means
sklean的包也有实现了k_means算法自己动手查阅
"""
import numpy as np


# Function: K Means
# -------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
def kmeans(X, k, maxIt):
    numPoints, numDim = X.shape

    dataSet = np.zeros((numPoints, numDim + 1))
    dataSet[:, :-1] = X

    # Initialize centroids randomly
    centroids = dataSet[np.random.randint(numPoints, size=k), :]
    centroids = dataSet[0:2, :]
    # Randomly assign labels to initial centorid
    #给中心点的最后一列赋值
    centroids[:, -1] = range(1, k + 1)

    # Initialize book keeping vars.
    iterations = 0
    oldCentroids = None

    # Run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        print "iteration: \n", iterations
        print "dataSet: \n", dataSet
        print "centroids: \n", centroids
        # Save old centroids for convergence test. Book keeping.
        #为了使oldCentroids与centroids是两部分所以要用np.copy如果改成oldCentroids = centroids则是一部分改变其中一个
        #另一个也会跟着改变不是我们想要的
        oldCentroids = np.copy(centroids)
        iterations += 1

        # Assign labels to each datapoint based on centroids
        #根据中心点根据数据集从新归类
        updateLabels(dataSet, centroids)

        # Assign centroids based on datapoint labels
        #归类完后重新算出中心点（根据更新后的数据集与K重新算出中心点）
        centroids = getCentroids(dataSet, k)

    # We can get the labels too by calling getLabels(dataSet, centroids)
    return dataSet


# Function: Should Stop
# -------------
# Returns True or False if k-means is done. K-means terminates either
# because it has run a maximum number of iterations OR the centroids
# stop changing.
#停止的2种情况1个是迭代次数大于阈值
def shouldStop(oldCentroids, centroids, iterations, maxIt):
    if iterations > maxIt:
        return True
    #2是比较新的中心点与旧的中心的是否一样如果一样就停止（比较2个值）
    return np.array_equal(oldCentroids, centroids)


# Function: Get Labels
# -------------
# Update a label for each piece of data in the dataset.
def updateLabels(dataSet, centroids):
    # For each element in the dataset, chose the closest centroid.
    # Make that centroid the element's label.
    numPoints, numDim = dataSet.shape
    for i in range(0, numPoints):
        #对当前行的最后一列进行归类
        dataSet[i, -1] = getLabelFromClosestCentroid(dataSet[i, :-1], centroids)


def getLabelFromClosestCentroid(dataSetRow, centroids):
    #初始化中心点的label
    label = centroids[0, -1];
    minDist = np.linalg.norm(dataSetRow - centroids[0, :-1])
    for i in range(1, centroids.shape[0]):
        #np.linalg.norm方法是传入两个向量算出两个向量之间的距离
        dist = np.linalg.norm(dataSetRow - centroids[i, :-1])
        if dist < minDist:
            #保证minDist是最小的
            minDist = dist
            label = centroids[i, -1]
    print "minDist:", minDist
    return label


# Function: Get Centroids
# -------------
# Returns k random centroids, each of dimension n.
#根据数据集包括标签的K是指分多少个类
def getCentroids(dataSet, k):
    # Each centroid is the geometric mean of the points that
    # have that centroid's label. Important: If a centroid is empty (no points have
    # that centroid's label) you should randomly re-initialize it.
    result = np.zeros((k, dataSet.shape[1]))
    for i in range(1, k + 1):
        oneCluster = dataSet[dataSet[:,-1] == i, :-1]
        #找出属于同一类i类后进行求均值axis=0是对行进行求均值axis=1是对所有的列进行求均值
        #给新的中心点的值进行赋值
        result[i - 1, :-1] = np.mean(oneCluster, axis=0)
        #赋值标签
        result[i - 1, -1] = i

    return result


x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 4])
#np.vstack方法是把四个点变成一个矩阵随机的
testX = np.vstack((x1, x2, x3, x4))

result = kmeans(testX, 2, 10)
print "final result:"
print result