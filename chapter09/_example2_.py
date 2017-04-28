# coding=utf-8
# 用梯度上升算法计算最佳回归系数，找函数最大值
# 画出决策边界
#
import logRegres

dataArr,labelMat = logRegres.loadDataSet()
weights = logRegres.gradAscent(dataArr,labelMat)
logRegres.plotBestFit(weights.getA())

