# coding=utf-8
# 二分切分数据代码
import regTrees
import numpy
testMat = numpy.mat(numpy.eye(4))
mat0,mat1 = regTrees.binSplitDataSet(testMat,1,0.5)
print testMat
print
print mat0
print
print mat1

