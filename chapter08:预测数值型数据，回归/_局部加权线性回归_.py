import regression
from numpy import *
import matplotlib.pyplot as plt

xArr, yArr = regression.loadDataSet('ex0.txt')

print(yArr[0])

xMat = mat(xArr)
srtInd = xMat[:,1].argsort(0) # argsort:从中可以看出argsort函数返回的是数组值从小到大的索引值
temp1 = xMat[srtInd] # 这里是个三维数组
xSort = xMat[srtInd][:,0,:]

yHat = regression.lwlr(xArr[0], xArr, yArr, 1.0)

yHat = regression.lwlr(xArr[0], xArr, yArr, 0.001)


#用测试数据预测
yHat = regression.lwlrTest(xArr, xArr, yArr, 1.0)
#画出预测的数值，与回归系数的线条
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[srtInd])
ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T[:,0].flatten().A[0], s = 2, c = 'red')
plt.show()

yHat = regression.lwlrTest(xArr, xArr, yArr, 0.01)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[srtInd])
ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T[:,0].flatten().A[0], s = 2, c = 'red')
plt.show()


yHat = regression.lwlrTest(xArr, xArr, yArr, 0.003)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[srtInd])
ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T[:,0].flatten().A[0], s = 2, c = 'red')
plt.show()
