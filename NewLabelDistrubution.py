#!/usr/bin/python
# _*_ coding:utf-8 _*_
# @Time   : 2019/6/26 0026 15:15
#@Author  :XCZ
#@File    :NewLabelDistrubution.py


import pandas as pd
import numpy as np
import copy
import math
import time


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


def GetTheSimiliarDegree(data,NEFeatures,NOFetures):
    """
    本函数是用于在标记增强之前计算对应的样本之间的相似度的
    本函数在提供的数据集中计算相互的两个样本在对应的特征子集下面的余弦相似度
    但是，在特征的集合中，可能存在连续的特征子集和离散的特征子集
    所以，对于离散的连续子集，我们将进行特定的数值处理
    :param data: 数据集
    :param NEFeatures:连续特征子集
    :param NOFetures: 离散特征子集
    :return: 相似矩阵
    """
    SimiliarMatrix = np.ones((len(data),len(data)))
    if NOFetures:
        CalcFeatures = copy.deepcopy(NEFeatures)
        CalcFeatures.extend(NOFetures)
        for x in range(0, len(data)-1):
            DataOfXToCalc = data[x][CalcFeatures]
            for y in range(x+1, len(data)):
                # 对于离散值的特定处理
                DataOfYToCalc = data[y][CalcFeatures]
                NOFeturesIndex = np.array(NOFetures)
                SameIndex = NOFeturesIndex[DataOfYToCalc[NOFetures] != DataOfXToCalc[NOFetures]].tolist()
                DiffIndex = list(set(NOFetures) - set(SameIndex))
                DataOfYToCalc[NOFetures] = 0
                DataOfXToCalc[SameIndex] = 0
                DataOfXToCalc[DiffIndex] = 1
                SimiliarDegree = (DataOfXToCalc.dot(DataOfYToCalc))/(np.linalg.norm(DataOfXToCalc)*np.linalg.norm(DataOfYToCalc))
                SimiliarMatrix[x][y] = SimiliarDegree
                SimiliarMatrix[y][x] = SimiliarDegree
    else:
        for x in range(len(data) - 1):
            for y in range(x+1, len(data)):
                SimiliarDegree = (data[x][NEFeatures].dot(data[y][NEFeatures]))/(np.linalg.norm(data[x][NEFeatures])*np.linalg.norm(data[y][NEFeatures]))
                SimiliarMatrix[x][y] = SimiliarDegree
                SimiliarMatrix[y][x] = SimiliarDegree
    return SimiliarMatrix


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
            SimiliarDegree = (data[x][Labels].dot(data[y][Labels])) / (np.linalg.norm(data[x][Labels]) * np.linalg.norm(data[y][Labels]))
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
        for i in labels:
            data[x][i] /= LabelSumOfX
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


def GetTheRelationMatrixOfAll(data, NEFeatures, NOFeatures, labels):
    """
    本方法中是用于构建对应的相似度矩阵的，
    其中单独的连续值特征和标记下面的样本相似性使用的是曼哈顿距离，
    离散值特征下面使用的是判断其是否相等
    所有标记之间使用的是余弦相似度
    :param data: 数据集
    :param NEFeatures: 连续型特征
    :param NOFeatures: 离散型特征
    :param labels: 标记集合
    :return: 关系矩阵
    """
    RelationMatrixofAll = np.ones((len(data),len(data),len(NEFeatures)+len(NOFeatures)+len(labels)+1))
    if NOFeatures:
        for x in range(len(data)-1):
            for y in range(x+1, len(data)):
                RelationOfLabels = (data[x][labels].dot(data[y][labels]))/(np.linalg.norm(data[x][labels])*np.linalg.norm(data[y][labels]))
                RelationMatrixofAll[x][y][-1] = RelationOfLabels
                RelationOfLabels = 1 - abs(data[x][labels] - data[y][labels])
                RelationOfAllNOFeatures = np.array([data[x][NOFeatures] == data[y][NOFeatures]]) + 0
                RelationOfAllNEFeatures = 1 - abs(data[x][NEFeatures] - data[y][NEFeatures])
                RelationMatrixofAll[x][y][NOFeatures] = RelationOfAllNOFeatures
                RelationMatrixofAll[x][y][NEFeatures] = RelationOfAllNEFeatures
                RelationMatrixofAll[x][y][labels] = RelationOfLabels
                RelationMatrixofAll[y][x] = copy.deepcopy(RelationMatrixofAll[x][y])
    else:
        for x in range(len(data) - 1):
            for y in range(x + 1, len(data)):
                RelationOfLabels = (data[x][labels].dot(data[y][labels])) / (np.linalg.norm(data[x][labels]) * np.linalg.norm(data[y][labels]))
                RelationMatrixofAll[x][y][-1] = RelationOfLabels
                RelationOfAll = 1 - abs(data[x] - data[y])
                RelationMatrixofAll[x][y][:-1] = RelationOfAll
                RelationMatrixofAll[y][x] = copy.deepcopy(RelationMatrixofAll[x][y])
    return RelationMatrixofAll


def GetTheFuzzyShannonEntropy(RelationMatrixOfAll, Subset):
    """
    本函数用于计算对应的模糊信息熵
    首先先判断提供的特征子集是某个单独的特征还是特征子集
    然后计算对应的关系矩阵
    接着按照对应的模糊信息熵公式计算对应的信息熵
    :param RelationMatrixOfAll:包含所有特征和标记下的关系矩阵
    :param Subset:计算信息熵时关联的特征子集（可能是某个特征，可能是某个标记，也有可能是某个特征子集）
    :return:返回一个float数值
    """
    RelationMatrixOfSubSet = np.ones((len(data),len(data)))
    if isinstance(Subset, int):
        RelationMatrixOfSubSet = copy.deepcopy(RelationMatrixOfAll[:,:,Subset])
    elif len(Subset)==1:
        RelationMatrixOfSubSet = copy.deepcopy(RelationMatrixOfAll[:, :, Subset])
    else:
        RelationMatrixOfSubSet = copy.deepcopy(np.min(RelationMatrixOfAll[:,:,Subset], axis=2))
    dataSize = len(RelationMatrixOfAll)
    FuzzyShannonEntropy = -(sum([math.log(np.sum(x)/dataSize) for x in RelationMatrixOfSubSet]))/dataSize
    return FuzzyShannonEntropy


def GetFuzzyJointEntropy(RelationMatrixOfAll, Subset, Label):
    """
    本函数用于计算特征子集和标记之间的联合熵
    本文中的重点还是构建对应的关系矩阵，结合特征子集和标记共同考虑，将获得对应的最小特征子集
    :param RelationMatrixOfAll: 关系矩阵
    :param Subset: 特征子集
    :param label:指定的标记
    :return: 返回联合熵数值
    """
    if isinstance(Subset, int):
        RelationMatrixOfSubSetAndLabel = copy.deepcopy(np.min(RelationMatrixOfAll[:,:,[Subset,Label]], axis=2))
    else:
        T = copy.deepcopy(Subset)
        T.extends(Label)
        RelationMatrixOfSubSetAndLabel = copy.deepcopy(np.min(RelationMatrixOfAll[:, :, T], axis=2))
    dataSize = len(RelationMatrixOfAll)
    FuzzyJointEntropy = -(sum([math.log(np.sum(x) / dataSize) for x in RelationMatrixOfSubSetAndLabel])) / dataSize
    return FuzzyJointEntropy


def GetFuzzySymmetricUncertainty(RelationMatrixOfAll, Subset, Label):
    """
    本函数用于计算对应的对称不确定性，首先，先计算在特征子集和标记下面的信息熵和联合熵，
    通过这些，计算相应的信息增益，然后再通过此函数计算对应的对称不确定性
    :param RelationMatrixOfAll:关系矩阵
    :param Subset:特征子集
    :param label:标记
    :return:对称不确定性数值
    """
    ShannonEntropyOfSubset = GetTheFuzzyShannonEntropy(RelationMatrixOfAll, Subset)
    ShannonEntropyOfLabel = GetTheFuzzyShannonEntropy(RelationMatrixOfAll, Label)
    JointEntropyOfSubsetAndLabel = GetFuzzyJointEntropy(RelationMatrixOfAll, Subset, Label)
    InformationGain = ShannonEntropyOfSubset + ShannonEntropyOfLabel - JointEntropyOfSubsetAndLabel
    FuzzySymmetricUncertainty = InformationGain/(ShannonEntropyOfSubset + ShannonEntropyOfLabel) * 2
    return FuzzySymmetricUncertainty


def GetFuzzyInformationRelavance(RelationMatrixOfAll, Subset, Labels):
    """
    在本函数中，我们将考虑特征子集和每个标记之间的对称不确定性和所有的标记之间的对称不确定性
    依此来考虑特征子集的重要度
    :param RelationMatrixOfAll:关系矩阵
    :param Subset: 特征子集
    :param Labels:标记的集合
    :return:
    """
    SymmetricUncertaintyOfAllLabel = GetFuzzySymmetricUncertainty(RelationMatrixOfAll, Subset, -1)
    print("已经计算完与总标记之间的关系")
    InformationRelavance = SymmetricUncertaintyOfAllLabel
    return InformationRelavance
    # return  SymmetricUncertaintyOfAllLabel


def Redection(RelationMatrixOfAll, NEFeatures, NOFeatures, Labels):
    """
    本函数，我们将结合上面的函数来获取属性重要度的排序
    1、单纯考虑特征和标记之间的重要度
    2、考虑特征和之前的标记的关联性
    :param RelationMatrixOfAll:关系矩阵
    :param conditions:
    :param Labels:
    :return:
    """
    S = copy.deepcopy(NEFeatures)
    S.extend(NOFeatures)
    # 定义一个列表，用于保存每个特征和标记之间的相关性的值的大小
    InformationOfConditions = np.zeros((len(S)))
    print("开始计算特征重要度")
    for a in S:
        InformationOfConditions[a] = GetFuzzyInformationRelavance(RelationMatrixOfAll, a, Labels)
        print("特征",a,"已被计算")
    ConditionsRank = np.argsort(-InformationOfConditions).tolist()
    return ConditionsRank


if __name__ == '__main__':
    dataFrame = pd.read_csv("data/Scene.csv",header=None)
    AllFeaturesAndLabels = dataFrame.columns.tolist()
    AllLabels = AllFeaturesAndLabels[-6:]
    AllFeatures = AllFeaturesAndLabels[:-6]
    NonimalFeatures = []
    NumericFeatures = list(set(AllFeatures) - set(NonimalFeatures))
    data = np.array(dataFrame,dtype='float')
    normaliza(data, NumericFeatures)
    RecordTime1 = time.time()
    theSimiliarMatrix = GetTheSimiliarDegreeOfLabels(data, AllLabels)
    GetTheLabelDistrubution(theSimiliarMatrix, data, AllLabels)
    print('标记分布已完成，耗时=', time.time()-RecordTime1)
    RecordTime2 = time.time()
    theRelationMatrixOfAll = GetTheRelationMatrixOfAll(data, NumericFeatures, NonimalFeatures, AllLabels)
    print('模糊关系矩阵构建完毕，耗时=', time.time() - RecordTime2)
    theConditionRank = Redection(theRelationMatrixOfAll, NumericFeatures, NonimalFeatures, AllLabels)
