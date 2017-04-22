# coding=utf-8
# 用随机梯度上升算法计算最佳回归系数，找函数最大值
# 画出决策边界
#
import logRegres
import numpy

dataArr,labelMat = logRegres.loadDataSet()
weights = logRegres.stocGradAscent0(numpy.array(dataArr), labelMat)
logRegres.plotBestFit(weights)

