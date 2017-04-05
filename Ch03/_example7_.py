# coding=utf-8
# 决策树的存储
import trees
myDat,labels = trees.createDataSet()
myTree = trees.createTree(myDat,labels) # 创建模型
print myTree
trees.storeTree(myTree,'classifierStorage.txt')
print trees.grabTree('classifierStorage.txt')

