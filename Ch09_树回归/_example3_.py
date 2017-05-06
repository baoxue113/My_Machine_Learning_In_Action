# coding=utf-8
# 将CART算法用于回归（使用数据ex0.txt）
import regTrees
from numpy import *
myDat = regTrees.loadDataSet('ex0.txt')
myMat = mat(myDat)
print regTrees.createTree(myMat)

