# coding=utf-8
# 选择最好的方式划分数据集
import trees
myDat,labels = trees.createDataSet()
print trees.chooseBestFeatureToSplit(myDat)


