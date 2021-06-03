import numpy as np
import pandas as pd
import math
import operator


def calcDist(p, clusterPoint):
    """
    计算欧式距离
    :param p: 元素点
    :param clusterPoint: 簇心点
    :return: 欧式距离
    """
    d = 0
    for i, j in zip(p, clusterPoint):
        d += math.pow(i - j, 2)

    return int(math.sqrt(d) * 100) / 100


class KMeans:
    """
        KMeans类：
        聚类操作

        :param data:
            用于分析的原始数据集

        :param clusterNumber:
            聚类的簇心个数

        :param clusterPoints: 可选
            在原始数据中指定初始簇心
    """
    def __init__(self,
                 data,
                 clusterNumber,
                 clusterPoints=None
                 ):
        #类型检测
        if not isinstance(clusterNumber, int):
            raise Exception(f"type of clusterNumber should be int instead of {type(clusterNumber)}")
        if isinstance(data, np.ndarray):
            data = data.tolist()
        elif not isinstance(data, list):
            raise Exception(f"type of data should be list instead of {type(data)}")

        if isinstance(clusterPoints, list):
            if len(clusterPoints) != clusterNumber:
                raise Exception(f"clusterNumber is {clusterNumber}, but only got {len(clusterPoints)} in list clusterPoints")
        elif clusterPoints is not None:
            raise Exception(f"specified clusterPoints should be list instead of {type(clusterPoints)}")

        self.clusterNumber = clusterNumber  #聚类簇心个数
        self.data = data                    #用于分析的数据集
        self.propertyNumber = len(data[0])  #属性个数，即特征个数
        self.distList = []                  #距离列表用于存放每个点到每个簇心的距离
        self.clusterPointSet = []           #簇心集合
        self.statusList = []                #每个点的聚类情况
        self.statusCount = []               #统计每种聚类的成员个数
        self.calcNewClusterList = []        #用于计算新簇心列表
        self.clusterPointChanged = False    #每一轮簇心是否发生改变
        self.stepsCount = 0                 #轮数统计

        if clusterPoints is None:
            while True:
                clusterPoints = np.random.randint(0, len(self.data), self.clusterNumber)    #随机初始簇心
                if len(np.unique(clusterPoints)) == len(clusterPoints):     #保证随机点不重复
                    break

        for i in clusterPoints:     #放入初始化簇心
            self.clusterPointSet.append(self.data[i])

    def step(self):
        self.distList.clear()   #每个点到簇心距离表清空
        self.calcNewClusterList = np.zeros((self.clusterNumber, self.propertyNumber)).tolist()  #初始化簇心计算表
        self.statusCount = [0] * self.clusterNumber #初始化聚类个数统计表

        for i in self.data:     #遍历每个点
            point_dist = []     #一个点到所有簇心距离
            for j in self.clusterPointSet:      #遍历计算一个点到所有簇心距离
                point_dist.append(calcDist(i, j))

            self.distList.append(point_dist)    #放入距离列表

        initial_round_flag = True if not len(self.statusList) else False    #判断是否为首次进入
        self.clusterPointChanged = False
        for i, c in enumerate(self.distList):
            min_index, min_value = min(enumerate(c), key=operator.itemgetter(1))    #得到一点到所有簇心中距离最短的项
            if not initial_round_flag:
                if self.statusList[i] is not min_index and not self.clusterPointChanged:    #比较该点当前最短簇心是否较上一轮发生变化
                    self.clusterPointChanged = True
            else:
                self.clusterPointChanged = True

            self.statusList.append(min_index)   #将该点的聚类记入
            self.statusCount[min_index] += 1    #统计该类的个数加1

            for j in range(self.propertyNumber):    #把该点的所有属性（特征）加到计算新簇心列表
                self.calcNewClusterList[min_index][j] += self.data[i][j]

        if not initial_round_flag:      #不是第一轮则将上一轮的聚类记录全删掉，只保留最新的聚类状况
            for i in range(len(self.data)):
                del self.statusList[0]

        for i, v1 in enumerate(self.calcNewClusterList):    #计算新的簇心
            for j, v2 in enumerate(v1):     #每种聚类总值除以该聚类成员个数
                v2 /= self.statusCount[i]
                self.calcNewClusterList[i][j] = int(v2 * 100) / 100     #保留两位小数

        self.clusterPointSet.clear()    #清空之前的簇心集合
        for i in self.calcNewClusterList:   #放入新的簇心集合
            self.clusterPointSet.append(i)

        self.stepsCount += 1            #轮数加1


if __name__ == "__main__":
    # data = np.array([[1, 2], [1, 4], [3, 1], [3, 5], [5, 2], [5, 4]]) - 简单测试用例
    # k = KMeans(data, 2, clusterPoints=[2, 3]) - 简单测试用例

    data = pd.read_csv("./测试集.data", header=None).values.tolist()
    k = KMeans(data, 2)

    while True:
        k.step()

        if not k.clusterPointChanged:
            break

    print(k.statusList, k.stepsCount)


