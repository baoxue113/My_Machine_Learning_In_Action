# coding=utf-8
from numpy import *
import adaboost
#datMat：训练数据，classLabels：标识数据
datMat,classLabels = adaboost.loadSimpData() # 加载数据
D = mat(ones((5,1))/5) #
print(adaboost.buildStump(datMat,classLabels,D))

