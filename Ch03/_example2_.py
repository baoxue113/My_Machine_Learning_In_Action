# coding=utf-8
# 划分数据集
import trees
myDat,labels = trees.createDataSet()
# myDat:待划分的数据集，第二个参数：代表选择第几列（也就是特征），第三个参数：用来划分的值
print trees.splitDataSet(myDat,0,1)
print trees.splitDataSet(myDat,0,0)


