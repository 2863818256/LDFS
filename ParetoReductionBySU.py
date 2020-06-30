#!/usr/bin/python
# _*_ coding:utf-8 _*_
# @Time   : 2019/5/20 15:09
#@Author  :XCZ
#@File    :ParetoReductionBySU.py
import numpy as np
import pandas as pd
import math
import time
import copy

#归一化
def Normalized(data,conditions):
    for i in conditions:
        maxValue = np.max(data[:,i])
        minValue = np.min(data[:,i])
        for j in range(len(data)):
            data[j][i] = 1.0*(data[j][i] - minValue)/(1.0*(maxValue - minValue))

# 根据特定的函数计算标记增强时的邻域的阈值
def CalculateTheNeighborhoodThreshold(data,omega):
    sum = 0
    for i in range(len(data[0])):
        sum += np.std(data[:,i])
    sum = sum/len(data[0])
    thres = sum/math.pow(omega,2)
    return thres

# 根据所得的阈值计算对应的邻域
def GetNeigborhoodMatrix(data,therdshold,conditions):
    NeigM = [[] for i in data]
    for i in range(len(data)):
        for j in range(len(data)):
            sign = 0
            if i == j:
                sign = 1
            else:
                sum = np.linalg.norm(data[i][conditions] - data[j][conditions])
                if sum <= therdshold:
                    sign = 1
            if sign == 1:
                NeigM[i].append(j)
    return NeigM

# 通过邻域的概念将其中的标记转变成为对应的标记分布
def ConvertIntoLabelDistrubution(data,NeighborhoodMatrix,labels):
    Record = [[] for i in data]
    for i in range(len(data)):
        Record[i] = [(int)(np.sum(data[NeighborhoodMatrix[i],j])) if data[i][j] == 1 else 0 for j in labels ]
    for i in range(len(data)):
        LabelSum = np.sum(Record[i])
        data[i,labels] = Record[i]/LabelSum

# 此处的关系是计算样本之间每个特征和每个标记之间的相似性关系
def GetRelationMatrix(data):
    RelationMatrix = np.zeros((len(data),len(data),len(data[0])))
    for x in range(len(data)):
        for y in range(x+1,len(data)):
            for i in range(len(data[0])):
                RelationMatrix[x][y][i] = 1 - math.fabs(data[x][i] - data[y][i])
            RelationMatrix[y][x] = RelationMatrix[x][y]
    return RelationMatrix
# 构建香农熵
def CalcShannonEntropy(RelationMatrix,feature):
    RelationOfFeature = RelationMatrix[:,:,feature]
    Entropy = 0
    n = len(RelationOfFeature)
    for i in range(len(RelationOfFeature)):
        Entropy -= math.log2((np.sum(RelationOfFeature[i]) / n))
    Entropy = Entropy / n
    return Entropy

def GetRelationJoin(RelationMatrix,feature1,feature2):
    RelationOfJoin = np.zeros((len(RelationMatrix),len(RelationMatrix)))
    RelationOfNow = RelationMatrix[:,:,[feature1,feature2]]
    for x in range(len(data)):
        for y in range(x+1,len(data)):
            RelationOfJoin[x][y] = np.min(RelationOfNow[x,y])
            RelationOfJoin[y][x] = RelationOfJoin[x][y]
    return RelationOfJoin

# 构建关联熵
def CalcJoinEntropy(RelationMatrix,feature1,feature2):
    theRelationOfJoin = GetRelationJoin(RelationMatrix,feature1,feature2)
    Entropy = 0
    n = len(theRelationOfJoin)
    for i in range(len(theRelationOfJoin)):
        Entropy -= math.log2((np.sum(theRelationOfJoin[i]) / n))
    Entropy = Entropy / n
    return Entropy

#构建对称不确定性
def CalcSymmetriUncertainty(RelationMatrix,feature1,feature2):
    Entropy1 = CalcShannonEntropy(RelationMatrix,feature1)
    Entropy2 = CalcShannonEntropy(RelationMatrix,feature2)
    EntropyOfJoin = CalcJoinEntropy(RelationMatrix,feature1,feature2)
    SU = (Entropy1+Entropy2-EntropyOfJoin)/(Entropy1 + Entropy2)
    return SU

#计算所有的特征和标记的对称不确定性
def CalcAllSU(RelationMatrix,conditions,decisions):
    MatrixOfSU = np.zeros((len(conditions),len(decisions)))
    labelIndex = len(conditions)
    for x in conditions:
        for y in decisions:
            MatrixOfSU[x][y-labelIndex] = CalcSymmetriUncertainty(RelationMatrix,x,y)
            print('特征',x,'完成计算')
    return MatrixOfSU

def CalcTheMostImportantFeature(MatrixOfSU,conditions):
    labelindex = [0 for x in range(len(conditions))]
    for x in conditions:
        if labelindex[x] == -1:
            continue
        for y in range(x + 1, len(conditions)):
            calc = (MatrixOfSU[x] - MatrixOfSU[y])
            c = set(((calc > 0) + 0) - ((calc < 0) + 0))
            if 1 in c:
                if -1 not in c:
                    labelindex[y] = -1
            else:
                labelindex[x] = -1
                break
    return labelindex

if __name__ == '__main__':
    StartTime = time.time()
    dataFrame = pd.read_csv('scence.csv', header=None)
    features = dataFrame.columns.tolist()
    labels = features[-6:]
    conditions = features[:-6]
    data = np.array(dataFrame)
    Normalized(data, conditions)
    Recordtime1 = time.time()
    NeighborhoodThreshold = CalculateTheNeighborhoodThreshold(data, 0.4)
    NeighborhoodMatrix = GetNeigborhoodMatrix(data, NeighborhoodThreshold, conditions)
    ConvertIntoLabelDistrubution(data, NeighborhoodMatrix, labels)  # 标记增强
    print('标记增强完成，耗时=', (time.time() - Recordtime1))
    #构建相似矩阵
    Recordtime2 = time.time()
    theRelationMatrix = GetRelationMatrix(data)
    print('相似性矩阵构建完成，耗时=',time.time()-Recordtime2)
    #构建特征关于标记的对称不确定性矩阵
    Recordtime3 = time.time()
    theMatrixOfSU = CalcAllSU(theRelationMatrix,conditions,labels)
    print('对称不确定性矩阵构成，耗时=',time.time()-Recordtime3)
