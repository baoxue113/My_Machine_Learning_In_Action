# coding=utf-8
from numpy import *
import adaboost
# datMat：训练数据，classLabels：标识数据
datMat,classLabels = adaboost.loadDataSet('horseColicTraining2.txt') # 加载数据
# 用训练数据训练分类器
classifierArray = adaboost.adaBoostTrainDS(datMat,classLabels,40) # 循环次数提升的话，能提高分类效果
# 加载测试数据
testArr,testLabelArr = adaboost.loadDataSet('horseColicTest2.txt')
# 用分类器预测测试数据
prediction10 = adaboost.adaClassify(testArr,classifierArray)
errArr = mat(ones((67,1)))
#print(errArr)
# 计算错误率
print(errArr[prediction10 != mat(testLabelArr).T].sum() / 67)

