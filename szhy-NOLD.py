#!/usr/bin/python
# _*_ coding:utf-8 _*_
# @Time   : 2019/8/23 0023 16:40
#@Author  :XCZ
#@File    :szhy-NOLD.py
import pandas as pd
import numpy as np
import time
import copy

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
            if (x1 == 0 or y1 == 0 or z1 == 0):
                SimiliarDegree = 0
            else:
                SimiliarDegree = z1 / (x1 * y1)
            SimiliarMatrix[x][y] = SimiliarDegree
            SimiliarMatrix[y][x] = SimiliarDegree
    return SimiliarMatrix

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
            else:
                DiscernOfSample /= maxOfDiscern
            for i in range(21):
                DiscernMatrix = DiscernOfSample * (DiscernOfSample>=(i*0.1/2))
                MaxDiscernibilityMatrix[i] += DiscernMatrix * (1-RelationMatrixOfLabels[x][y])
    for i in range(21):
        FeatureRanks[i] = (np.argsort(-MaxDiscernibilityMatrix[i])+1) #将每个特征计算出来的标记和权重衡量之和进行计算，得到的结果进行由大到小的排序
    return FeatureRanks.tolist()


if __name__ == '__main__':
    StartTime = time.time()
    DataSetName = 'CHD_49'
    print('执行数据集', DataSetName)
    dataFrame = pd.read_csv('data/' + DataSetName + '.csv', header=None)
    features = dataFrame.columns.tolist()
    AllLabels = features[-6:]
    AllFeatures = features[:-6]
    data = np.array(dataFrame)
    NonimalFeatures = []
    NumericFeatures = list(set(AllFeatures) - set(NonimalFeatures))
    data = np.array(dataFrame, dtype='float')
    normaliza(data, NumericFeatures)
    Recordtime1 = time.time()
    theRelationMatrixOfLabels = GetTheSimiliarDegreeOfLabels(data, AllLabels)
    theReduction = GetTheSamplePairsMaxDiscernibility(data, NumericFeatures, NonimalFeatures,theRelationMatrixOfLabels)
    with open('ExperimentResult/NOLD/'+DataSetName+'_Result.txt','w') as fw:
        for i in range(len(theReduction)):
            fw.write("ItemList["+str(i)+"]="+str(theReduction[i])+"\n")
    print("约简用时：",time.time()-Recordtime1)

