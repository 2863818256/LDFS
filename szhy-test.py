#!/usr/bin/python
# _*_ coding:utf-8 _*_
# @Time   : 2019/11/12 0012 11:30
#@Author  :XCZ
#@File    :szhy-test.py
import pandas as pd
import numpy as np
import time
import copy
import math
import os,sys
import random

def normaliza(data, conditions):
    """
    此函数是用于将导入的数据集中的特征值进行规范化的
    我们首先将我们获得的数据导入到此函数中，
    然后对于每个特征下面的数值，我们将其按照线性规则等比例缩放

    :param data: 这是我们导入的数据集
    :param conditions: 这是数据集中的连续值的特征的集合
    :return: 无返回值
    """
    for i in conditions:
        max = np.max(data[:,i])
        min = np.min(data[:,i])
        for j in range(len(data)):
            data[j][i] = 1.0*(data[j][i] - min)/(1.0*(max - min))

def getTheDistanceOfSample(data, index, NOFeatures, NEFeatures):
    """
     计算选中的样本index和其他样本之间的距离
    :param data: 数据集
    :param index: 选中的样本索引
    :param NOFeatures: 离散型数据
    :param NEFeatures: 连续型数据
    :return: 样本和其他样本之间的距离
    """
    theDistanceArray = np.zeros((len(data)))
    for i in range(len(data)):
        theDistanceOfFeature = np.zeros((len(NOFeatures)+len(NEFeatures)))
        if i == index:
            continue
        if NOFeatures:
            theDistanceOfFeature[NOFeatures] = data[index][NOFeatures]!=data[i][NOFeatures] + 0
        theDistanceOfFeature[NEFeatures] = np.square(data[index][NEFeatures] - data[i][NEFeatures])
        theDistanceArray[i] = math.sqrt(np.sum(theDistanceOfFeature))
    return theDistanceArray

# def GetTheSimiliarDegree(data,NOFeatures,NEFeatures):
#     """
#     本函数是用于在标记增强之前计算对应的样本之间的相似度的
#     本函数在提供的数据集中计算相互的两个样本在对应的特征子集下面的余弦相似度
#     但是，在特征的集合中，可能存在连续的特征子集和离散的特征子集
#     所以，对于离散的连续子集，我们将进行特定的数值处理
#     :param data: 数据集
#     :param NEFeatures:连续特征子集
#     :param NOFeatures: 离散特征子集
#     :return: 相似矩阵
#     """
#     theSimiliarMatrix = np.zeros((len(data),len(data)))
#     for x in range(len(data)):
#         theDiffernceOfX = getTheDistanceOfSample(data, x, NOFeatures,NEFeatures)
#         theRelativeDifference = theDiffernceOfX/np.max(theDiffernceOfX)
#         theSimiliarMatrix[x] = 1 - theRelativeDifference
#         theSimiliarMatrix[x] = 1 - getTheDistanceOfSample(data, x, NOFeatures,NEFeatures)/len(data)
#     return theSimiliarMatrix

def GetTheSimiliarDegreeOfLabels(data, Labels):
    """
    本函数是用于实现上述函数的另一种情况，主要是用于使用标记来计算对应的相似度
    :param data:数据集
    :param Labels:标记集合
    :return:相似度矩阵
    """
    SimiliarMatrix = np.ones((len(data), len(data)))
    for x in range(len(data) - 1):
        for y in range(x + 1, len(data)):
            x1 = np.linalg.norm(data[x][Labels])
            y1 = np.linalg.norm(data[y][Labels])
            z1 = (data[x][Labels].dot(data[y][Labels]))
            if (x1 == 0 or y1 == 0):
                SimiliarDegree = 0
            else:
                SimiliarDegree = z1 / (x1 * y1)
            SimiliarMatrix[x][y] = SimiliarDegree
            SimiliarMatrix[y][x] = SimiliarDegree
    return SimiliarMatrix


def getTheDistancesOfSamples(data, NEfeature, NOfeature):
    """
    此函数用于计算样本之间的距离
    :param data: 数据集
    :param NEfeature: 连续型数据
    :param NOfeature: 离散型数据
    :return: 距离矩阵
    """
    DistanceOfSamples = np.zeros((len(data),len(data)))
    SimilarityOfSamples = np.ones((len(data),len(data)))
    for x in range(len(data)-1):
        for y in range(x+1,len(data)):
            theDistance = np.zeros((len(data[0])))
            if NOfeature:
                theDistance[NOfeature] = (data[x][NOfeature]!=data[y][NOfeature])+1
            theDistance[NEfeature] = np.abs(data[x][NEfeature] - data[y][NEfeature])
            theAllDistance = math.sqrt(np.sum(theDistance**2))
            DistanceOfSamples[x][y] = theAllDistance
            DistanceOfSamples[y][x] = theAllDistance
    for x in range(len(data)):
        DistanceOfSamples[x][x] = np.max(DistanceOfSamples[x])
        DistanceOfSamples[x][x] = np.min(DistanceOfSamples[x])
        DistanceOfSamples[x] = (DistanceOfSamples[x] - np.min(DistanceOfSamples[x]))/(np.max(DistanceOfSamples[x])-np.min(DistanceOfSamples[x]))
    return DistanceOfSamples

def GetTheSamplePairsMaxDiscernibility(data, NEfeature, NOfeature, RelationMatrixOfLabels):
    """
    在本函数下面计算每个样本下面的对应的最大可辨识特征
    :param data: 数据集
    :param NEfeature: 连续值特征
    :param NOfeature: 离散值特征
    :return: 可辨识对矩阵
    """
    S = copy.deepcopy(NEfeature)
    if NOfeature:
        S.extend(NOfeature)
    MaxDiscernibilityMatrix = np.zeros((21,len(S)))
    RelationOfSample = np.zeros((len(S)))
    indexLen = len(S)
    FeatureRanks = np.zeros((21,len(S)),dtype='int')
    for x in range(len(data)):
        for y in range(len(data)):
            if NOfeature:
                RelationOfSample[NOfeature] = data[x][NOfeature]==data[y][NOfeature] + 0
            RelationOfSample[NEfeature] = 1 - abs(data[x][NEfeature] - data[y][NEfeature])   #计算相似度
            DiscernOfSample = 1 - RelationOfSample
            maxOfDiscern = np.max(DiscernOfSample)
            if maxOfDiscern == 0:
                DiscernOfSample += 1/(indexLen)
            for i in range(21):
                DiscernMatrix = DiscernOfSample * ((DiscernOfSample/np.max(DiscernOfSample))>=(i*0.1/2))
                MaxDiscernibilityMatrix[i] += DiscernMatrix * (1-RelationMatrixOfLabels[x][y])
    for i in range(21):
        FeatureRanks[i] = (np.argsort(-MaxDiscernibilityMatrix[i])+1) #将每个特征计算出来的标记和权重衡量之和进行计算，得到的结果进行由大到小的排序
    return FeatureRanks.tolist()


if __name__ == '__main__':
    StartTime = time.time()
    DataSetName = 'EukaryotePseAAC'
    print('执行数据集', DataSetName)
    dataFrame = pd.read_csv('data/' + DataSetName + '.csv', header=None)
    features = dataFrame.columns.tolist()
    AllLabels = features[-22:]
    AllFeatures = features[:-22]
    data = np.array(dataFrame)
    NonimalFeatures = []
    NumericFeatures = list(set(AllFeatures) - set(NonimalFeatures))
    data = np.array(dataFrame, dtype='float')
    normaliza(data, NumericFeatures)
    Recordtime1 = time.time()
    theRelationMatrixOfLabels = GetTheSimiliarDegreeOfLabels(data, AllLabels)
    theReduction = GetTheSamplePairsMaxDiscernibility(data, NumericFeatures, NonimalFeatures, theRelationMatrixOfLabels)
    with open('ExperimentResult/NOLD1/' + DataSetName + '_Result.txt', 'w') as fw:
        for i in range(len(theReduction)):
            fw.write("ItemList[" + str(i) + "]=" + str(theReduction[i]) + "\n")
    print("约简用时：", time.time() - Recordtime1)
