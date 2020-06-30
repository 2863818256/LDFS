#!/usr/bin/python
# _*_ coding:utf-8 _*_
# @Time   : 2019/9/25 0025 19:53
#@Author  :XCZ
#@File    :szhy-LD.py
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


def GetTheSimiliarDegree(data,NOFeatures,NEFeatures):
    """
    本函数是用于在标记增强之前计算对应的样本之间的相似度的
    本函数在提供的数据集中计算相互的两个样本在对应的特征子集下面的余弦相似度
    但是，在特征的集合中，可能存在连续的特征子集和离散的特征子集
    所以，对于离散的连续子集，我们将进行特定的数值处理
    :param data: 数据集
    :param NEFeatures:连续特征子集
    :param NOFeatures: 离散特征子集
    :return: 相似矩阵
    """
    SimiliarMatrix = np.ones((len(data),len(data)))
    if NOFeatures:
        CalcFeatures = copy.deepcopy(NEFeatures)
        CalcFeatures.extend(NOFeatures)
        for x in range(0, len(data)-1):
            DataOfXToCalc = data[x][CalcFeatures]
            for y in range(x+1, len(data)):
                # 对于离散值的特定处理
                DataOfYToCalc = data[y][CalcFeatures]
                NOFeturesIndex = np.array(NOFeatures)
                SameIndex = NOFeturesIndex[DataOfYToCalc[NOFeatures] != DataOfXToCalc[NOFeatures]].tolist()
                DiffIndex = list(set(NOFeatures) - set(SameIndex))
                DataOfYToCalc[NOFeatures] = 0
                DataOfXToCalc[SameIndex] = 0
                DataOfXToCalc[DiffIndex] = 1
                SimiliarDegree = (DataOfXToCalc.dot(DataOfYToCalc))/(np.linalg.norm(DataOfXToCalc)*np.linalg.norm(DataOfYToCalc))
                SimiliarMatrix[x][y] = SimiliarDegree
                SimiliarMatrix[y][x] = SimiliarDegree
    else:
        for x in range(len(data) - 1):
            for y in range(x+1, len(data)):
                x1 = np.linalg.norm(data[x])
                y1 = np.linalg.norm(data[y])
                z1 = (data[x].dot(data[y]))
                if (x1==0 or y1==0 or z1==0):
                    SimiliarDegree = 0
                else:
                    SimiliarDegree = z1/(x1*y1)
                SimiliarMatrix[x][y] = SimiliarDegree
                SimiliarMatrix[y][x] = SimiliarDegree
    return SimiliarMatrix

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
            if (x1 == 0 or y1 == 0 or z1 == 0):
                SimiliarDegree = 0
            else:
                SimiliarDegree = z1 / (x1 * y1)
            SimiliarMatrix[x][y] = SimiliarDegree
            SimiliarMatrix[y][x] = SimiliarDegree
    return SimiliarMatrix

def GetTheLabelDistrubution(SimiliarMatrix,data,labels):
    """
    本函数是应用相应的相似度矩阵来运算对应的标记增强
    本函数的基础是对应样本的模糊相似度和每个标记下面的标记结果相乘得到对应的标记增强结果
    1、首先先计算两个矩阵的运算
    2、然后将同一个样本下面的标记增强结果进行归一化
    3、如果所有为0的标记的标记增强的和大于原先定义的标记增强的结果，就要将为0的标记增强结果进行缩放
    （此处的缩放方式有待考究，如果将原先的标记粗暴的按照缩放，本次标记增强的结果就没有意义）
    4、将缩放后的结果中过小的结果设置为0，避免生出误差
    5、将最后获得的结果进行标记再次进行归一化
    RaiserError:
    现在本方法存在一个大问题，即它可能会修改对应的特征归一化结果（出错了！！！！！）
    :param SimiliarMatrix: 相似度矩阵
    :param data: 需要进行的标记增强的对应数据集
    :param labels:对应数据集中的标记集合
    :return:无返回值
    """
    LabelInformation = copy.deepcopy(data[:,labels])
    #定义标记的起始位置
    labelstart = labels[0]
    SampelLabelCount = np.zeros((len(data)))
    # 将list的标记转变为array格式的标记
    LabelIndex = np.array(labels)
    #首先，通过相似矩阵和标记矩阵的相乘，得出对应的标记矩阵的初始增强结果
    for x in range(len(data)):
        for i in labels:
            data[x][i] = SimiliarMatrix[x].dot(LabelInformation[:,i-labelstart])
        #在得到结果之后，我们对每个样本的标记结果进行初始的标记增强的归一化
        LabelSumOfX = np.sum(data[x][labels])
        if LabelSumOfX == 0:
            data[x][labels] = 1/len(labels)
            continue
        data[x][labels] /= LabelSumOfX
    for x in range(len(data)):
        SampelLabelCount[x] = np.sum(LabelInformation[x])/(len(labels))
        ZeroLabelIndexOfX = LabelIndex[(LabelInformation[x] == 0) ]
        OneLabelIndexOfX = LabelIndex[(LabelInformation[x] == 1) ]
        # 如果所有为 0 的标记增强的值大于原先标记的结果，那么我们的标记应该再次进行一次缩放，设定公式保证其值不会超过一定范围
        if np.sum(data[x][ZeroLabelIndexOfX ]) > 1 - SampelLabelCount[x]:
            data[x][ZeroLabelIndexOfX] = data[x][ZeroLabelIndexOfX] * (1 - SampelLabelCount[x])
            RemainValue = 1 - np.sum(data[x][ZeroLabelIndexOfX])
            data[x][OneLabelIndexOfX] = data[x][OneLabelIndexOfX]/np.sum(data[x][OneLabelIndexOfX])*RemainValue
        LabelMaxOfX = np.max(data[x][labels])
        for i in labels:
            if data[x][i] / LabelMaxOfX < 0.25 :
                data[x][i] = 0
        LabelSumOfX = np.sum(data[x][labels])
        data[x][labels] /= LabelSumOfX


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
    DataSetName = 'EukaryotePseAAC'
    print('执行数据集',DataSetName)
    dataFrame = pd.read_csv('data/'+DataSetName+'.csv', header=None)
    features = dataFrame.columns.tolist()
    AllLabels = features[-22:]
    AllFeatures = features[:-22]
    data = np.array(dataFrame)
    NonimalFeatures = []
    NumericFeatures = list(set(AllFeatures) - set(NonimalFeatures))
    data = np.array(dataFrame, dtype='float')
    Recordtime1 = time.time()
    normaliza(data, NumericFeatures)
    # 使用模糊相似关系的标记增强方法
    theSimiliarMatrix = GetTheSimiliarDegreeOfLabels(data, AllLabels)
    # theSimiliarMatrix2 = GetTheSimiliarDegree(data, NonimalFeatures, NumericFeatures)
    GetTheLabelDistrubution(theSimiliarMatrix, data, AllLabels)
    print('标记增强完成，耗时=', (time.time() - Recordtime1))
    # 将标记增强的文件转换为对应的标记增强文件
    # if not os.path.exists('LabelDistrubuteFile/'+DataSetName+'_conditions.csv'):
    #     f = open("LabelDistrubuteFile/'+DataSetName+'_conditions.csv","w")
    #     f.close()
    pd.DataFrame(data).to_csv('LabelDistrubuteFile/'+DataSetName+'_conditions.csv')
    # 标记增强之后，计算各个标记之间的最大相似关系
    theRelationMatrixOfLabels = GetTheSimiliarDegreeOfLabels(data, AllLabels)
    theReduction = GetTheSamplePairsMaxDiscernibility(data, NumericFeatures, NonimalFeatures,theRelationMatrixOfLabels)
    with open('ExperimentResult/test/'+DataSetName+'_Result.txt','w') as fw:
        for i in range(len(theReduction)):
            fw.write("ItemList["+str(i)+"]="+str(theReduction[i])+"\n")
