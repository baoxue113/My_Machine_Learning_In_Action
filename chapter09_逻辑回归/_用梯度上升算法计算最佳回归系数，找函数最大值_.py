# coding=utf-8
# 用梯度上升算法计算最佳回归系数，找函数最大值
#
import logRegres

dataArr,labelMat = logRegres.loadDataSet()
print (logRegres.gradAscent(dataArr,labelMat))

