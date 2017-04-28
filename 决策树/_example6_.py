# coding=utf-8
# 使用决策树执行分类(容易理解)
import treePlotter
import trees
myDat, labels = trees.createDataSet()
print labels
myTree = treePlotter.retrieveTree(0)
print myTree # myTree : 模型，决策树
# 模型，标签，预测数据
print trees.classify(myTree, labels,[1,0])
print trees.classify(myTree, labels,[1,1])

