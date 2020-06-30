#!/usr/bin/python
# _*_ coding:utf-8 _*_
# @Time   : 2019/10/6 0006 9:23
#@Author  :XCZ
#@File    :ExperimentAction.py

import os,sys

# 现有的工作环境
Current_Path = os.getcwd()
# 实验数据所在的环境是
Experiment_Path = "C:/Users/xcz19/Desktop/实验/Experiment_result"
os.chdir(Experiment_Path)
# 填入相应数据集所在的文件夹
DataSet_Name = "Water-quality"
# 写入对应的数据集特征选择方法
Experiment_Ways = ["MDDMproj","MDDMspc","PMU","ReliefFML","MLFRSE"]
# 按照对应的特征选择方法，将对应的特征选择方法读出
for index in range(len(Experiment_Ways)):
    strs = ''
    with open(DataSet_Name+'/'+DataSet_Name+"_"+Experiment_Ways[index]+".txt",'r') as fr:
        strs = fr.read().replace("\n",'')
        if (strs[-1] == '\n'):
             print(index)
        print("ItemList[" + str(index) + "]=",strs)
os.chdir(Current_Path)