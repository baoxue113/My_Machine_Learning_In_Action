# coding=utf-8
# 将CART算法用于回归 用的是后剪枝技术
import regTrees
from numpy import *

myDat2 = regTrees.loadDataSet('ex2.txt')
myMat2 = mat(myDat2)
# 创建最大的树
myTree = regTrees.createTree(myMat2,ops=(0,1))
# 导入测试数据
myDatTest = regTrees.loadDataSet('ex2test.txt')
myMat2Test = mat(myDatTest)
print myTree
print regTrees.prune(myTree, myMat2Test)