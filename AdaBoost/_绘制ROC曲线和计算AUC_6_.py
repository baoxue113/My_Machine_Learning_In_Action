# coding=utf-8
from numpy import *
import adaboost
# datMat：训练数据，classLabels：标识数据
datMat,classLabels = adaboost.loadDataSet('horseColicTraining2.txt') # 加载数据
# 用训练数据训练分类器
classifierArray,aggClassEst = adaboost.adaBoostTrainDS(datMat,classLabels,10) # 循环次数提升的话，能提高分类效果
adaboost.plotROC(aggClassEst.T,classLabels)

# Backend MacOSX is interactive backend. Turning interactive mode on.
# the Area Under the Curve is:  0.8582969635063604 ，正确率刚好80%


