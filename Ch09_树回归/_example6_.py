# coding=utf-8
# 模型树
import regTrees
from numpy import *
myDat2 = regTrees.loadDataSet('exp2.txt')
myMat2 = mat(myDat2)
print regTrees.createTree(myMat2,regTrees.modelLeaf,regTrees.modelErr,(1,10))
