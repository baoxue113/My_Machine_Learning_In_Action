# coding=utf-8
# 二分切分数据代码
import regTrees
from numpy import *
myDat = regTrees.loadDataSet('ex00.txt')
myMat = mat(myDat)
print regTrees.createTree(myMat)

