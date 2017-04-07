# coding=utf-8
# 树回归于标准回归的比较
import regTrees
from numpy import *
trainMat = mat(regTrees.loadDataSet('bikeSpeedVsIq_train.txt'))
testMat = mat(regTrees.loadDataSet('bikeSpeedVsIq_test.txt'))
myTree = regTrees.createTree(trainMat,ops = (1,20)) # 构建一个回归树
yHat = regTrees.createForeCast(myTree,testMat[:,0]) # 预测结果
temp1 = corrcoef(yHat,testMat[:,1],rowvar=0)
print corrcoef(yHat,testMat[:,1],rowvar=0)[0,1] # 计算模型树、回归树及其他模型效果，比较客观的方法是计算相关系数，R*R值，Numpy中corrcoef(yHat, y, rowvar = 0)也即皮尔逊相关系数。
print

myTree = regTrees.createTree(trainMat, regTrees.modelLeaf, regTrees.modelErr, (1,20))# 构建一个模型树
yHat = regTrees.createForeCast(myTree, testMat[:,0],regTrees.modelTreeEval)
print corrcoef(yHat,testMat[:,1],rowvar=0)[0,1] # 计算皮尔逊相关系数
print

ws,X,Y = regTrees.linearSolve(trainMat) # 构建线性模型
print ws
print

for i in range(shape(testMat)[0]):
    temp2 = ws[1,0]
    temp3 = ws[0,0]
    yHat[i] = testMat[i,0] * ws[1,0] + ws[0,0] # 线性模型预测结果
print corrcoef(yHat,testMat[:,1],rowvar=0)[0,1] # 计算皮尔逊相关系数
print