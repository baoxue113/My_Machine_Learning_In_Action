# coding=utf-8
# 查看香浓信息熵的增益
import trees
myDat,labels = trees.createDataSet()
print trees.calcShannonEnt(myDat)
myDat[0][-1] = 'maybe'
print trees.calcShannonEnt(myDat)

