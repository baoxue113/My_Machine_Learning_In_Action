# coding=utf-8

# 归一化处理数据
from array import array

import kNN
import matplotlib
import matplotlib.pyplot as plt
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt') #数据预处理

normMat,ranges,minVals = kNN.autoNorm(datingDataMat)
print normMat
print ranges
print minVals