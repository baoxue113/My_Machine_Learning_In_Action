# coding=utf-8
from numpy import *
import adaboost
#datMat：训练数据，classLabels：标识数据
datMat,classLabels = adaboost.loadSimpData() # 加载数据
# 训练分类器
classifierArray = adaboost.adaBoostTrainDS(datMat,classLabels,9)
print(classifierArray)

