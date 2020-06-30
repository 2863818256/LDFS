#!/usr/bin/python
# _*_ coding:utf-8 _*_
# @Time   : 2018/12/23 15:11
#@Author  :XCZ
#@File    :GetAttribute.py
S = [18,4,67,23,61,13,57,64,70,26,10,1,36,6,71,5,34,55,9,58,39,40,49,54,17,14,51,22,62,50,65,16,25,33,66,8,56,31,53,29,35,37,47,11,38,27,42,24,2,69,59,41,28,72,7,12,45,63,32,48,44,68,19,46,60,15,30,20,21,43,52,3]
leng = 2
time = (int)(len(S)/leng+1)
for i in range(time):
    attribute = 'attribute['+str(i)+'] = "'
    if i==0:
        s = sorted(S)
    else:
        s = sorted(S[:-leng*i])
    c = 0
    for item in s:
        attribute = attribute+str(item)
        c = c + 1
        if c != len(s):
            attribute = attribute + ','
    attribute = attribute + '";'
    print(attribute)