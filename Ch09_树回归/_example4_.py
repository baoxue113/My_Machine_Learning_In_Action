# coding=utf-8
# 将CART算法用于回归 用的是预剪枝
import regTrees
from numpy import *
myDat = regTrees.loadDataSet('ex0.txt')
myMat = mat(myDat)
print regTrees.createTree(myMat,ops=(0,1))
print regTrees.createTree(myMat)

myDat2 = regTrees.loadDataSet('ex2.txt')
myMat2 = mat(myDat2)
print regTrees.createTree(myMat2)
print regTrees.createTree(myMat2,ops=(10000,4))