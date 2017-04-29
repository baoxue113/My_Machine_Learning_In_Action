# coding=utf-8
from numpy import *
import apriori
# 加载数据
dataSet = apriori.loadDataSet() # 加载数据
print(dataSet)
C1 = apriori.createC1(dataSet)
print(C1)
D = map(set,dataSet)
print(D)
L1,suppData0 = apriori.scanD(D,C1,0.5)
print(L1)

