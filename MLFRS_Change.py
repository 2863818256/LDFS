#!/usr/bin/python
# _*_ coding:utf-8 _*_
# @Time   : 2019/10/9 0009 21:49
#@Author  :XCZ
#@File    :MLFRS_Change.py
import numpy as np
Feature_num = 49
Feature_subset = [49, 44, 16, 26, 41, 18, 14, 12, 36, 5]
Feature_subset.extend(np.array(set(list(range(1,Feature_num+1)))-set(Feature_subset)).tolist())
print(Feature_subset)
