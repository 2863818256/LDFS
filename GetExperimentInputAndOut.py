#!/usr/bin/python
# _*_ coding:utf-8 _*_
# @Time   : 2019/9/10 0010 9:44
#@Author  :XCZ
#@File    :GetExperimentInputAndOut.py

ItemList = [ [] for i in range(5)]
S = [ [] for i in range(5)]
featureNum = 16
ItemList[0]= [10,13,2,15,1,16,4,7,9,14,11,8,6,5,12,3]
ItemList[1]= [4,6,1,10,7,2,9,8,13,11,12,15,3,16,14,5]
ItemList[2]= [7,13,9,14,1,12,11,15,6,3,5,8,10,4,2,16]
ItemList[3]= [12,13,7,8,3,6,9,5,4,14,11,15,16,10,2,1]
ItemList[4]= [1,2,13,7,8,6,4,12,11,14,9,16,3,5,10,15]
if len(ItemList[0])!=featureNum:
    print("输出错误")
if(featureNum<100):
    Size = round(featureNum*0.4)
elif(featureNum<500):
    Size = round(featureNum * 0.3)
elif(featureNum<1000):
    Size = round(featureNum * 0.2)
else:
    Size = round(featureNum * 0.1)
for i in range(len(ItemList)):
    S[i] = ItemList[i][:Size]
    attribute = 'attribute[' + str(i) + '] = "'
    s = sorted(S[i])
    c = 0
    for item in s:
        attribute += str(item)
        c += 1
        if c!= len(s):
            attribute += ','
    attribute += '";'
    print(attribute)