#!/usr/bin/python
# _*_ coding:utf-8 _*_
# @Time   : 2018/12/7 20:36
#@Author  :XCZ
#@File    :Simple.py
import pandas as pd
import numpy as np
import random
import math
from copy import deepcopy
import time

def normaliza(data,conditions):            #归一化
    for i in conditions:
        max = np.max(data[:,i])
        min = np.min(data[:,i])
        for j in range(len(data)):
            data[j][i] = 1.0*(data[j][i] - min)/(1.0*(max - min))

def findDiffernet(data,labels):
    DIFX = np.zeros((len(data),len(data)))
    dataL = data[:,labels]
    for x in range(len(data)):
        for y in range(x+1,len(data)):
            sum = 0
            for l in range(len(labels)):
                if dataL[x,l] != dataL[y,l]:
                    sum += 1
            DIFX[x,y] = sum
            DIFX[y,x] = sum
    return DIFX

def rank(data,labels):
    DIFX = findDiffernet(data,labels)
    rankD = [[] for i in DIFX]
    for i in range(len(DIFX)):
        s = np.argsort(-DIFX[i])
        for index in s:
            if DIFX[i,index] == 0:
                break
            rankD[i].append(index)
    return rankD

def Draw(rankD,size,proportion):
    s = [[] for i in rankD]
    for i in range(len(rankD)):
        lenght = (int)(len(rankD[i])/size)
        index = 0
        for d in range(len(proportion)):
            s[i].extend(random.sample(rankD[i][index:index+lenght],(int)(lenght*proportion[d])))
            index += lenght
    return s


def CalcRelationOfT(data,draw,ak,RelationOfB):
    RelationOfT = [[0 for j in range(len(draw[i]))] for i in range(len(data))]
    RelationOfAk = [[0 for j in range(len(draw[i]))] for i in range(len(data))]
    for i in range(len(data)):
        RelationOfAk[i] =(1 - abs(data[i,ak] - data[draw[i],ak])).tolist()
        for j in range(len(draw[i])):
            RelationOfT[i][j] = min(RelationOfB[i][j],RelationOfAk[i][j])
    return RelationOfT

def CalclowerApproximation(RelationOfT):
    lower = [0 for i in data]
    for i in range(len(data)):
        SumOfI = sum([1-item for item in RelationOfT[i]])
        lower[i] = SumOfI/(len(RelationOfT[i]))
    return lower

def CalcDependenceDegree(RelationOfT):
    L = CalclowerApproximation(RelationOfT)
    return sum(L)/(len(data))

def Sig(RelationOfT,ddB):
    ddT = CalcDependenceDegree(RelationOfT)
    return ddT - ddB


def Reduction(data, condition, labels, num):
    B = deepcopy(condition)
    red = []
    rankD = rank(data,labels)
    size = 3
    i = 0
    proportion = [0.015, 0.013, 0.012]
    theDraw = Draw(rankD,size,proportion)
    start = True
    theRelationOfReduction = [[ 1 for j in theDraw[i]] for i in range(len(data))]
    while start:
        value = 0
        index = -1
        theRelationOfReord = [[1 for j in theDraw[i]] for i in range(len(data))]
        stime = time.time()
        if not red:
            ddR = 0
        else:
            ddR = CalcDependenceDegree(theRelationOfReduction)
        for ak in B:
            theRelationOfT = CalcRelationOfT(data,theDraw,ak,theRelationOfReduction)
            s = Sig(theRelationOfT,ddR)
            if value < s:
                value = s
                index = ak
                theRelationOfReord = deepcopy(theRelationOfT)
        if index != -1:
            i= i+1
            print(value, "第%d个--用时：" % i, time.time() - stime)
            print(index)
            if value > 0:
                red.append(index)
                B.remove(index)
                theRelationOfReduction = deepcopy(theRelationOfReord)
                if len(red) == num:
                    return red
            else:
                start = False
        else:
            break
    redc = [item+1 for item in red]
    return redc


dataset = "CHD_49"
dataFrame = pd.read_csv('data/%s.csv' % dataset, header=None)
features = dataFrame.columns.tolist()
labels = features[-6:]
conditions = features[:-6]
data = dataFrame.values
normaliza(data, conditions)
start = time.time()
r = Reduction(data, conditions, labels, 200)
end = time.time()
print(end-start)
endTime = time.time()
print('runtime:', endTime - start)
