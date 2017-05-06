import regression
from numpy import *
xArr, yArr = regression.loadDataSet('ex0.txt')

print(xArr[0:2])

ws = regression.standRegres(xArr, yArr) # 求解回归系数
print(ws)

xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat * ws # 预测数据点的值

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])

xCopy = xMat.copy()
xCopy.sort(0) # 按值传递，所以copy,然后排序
yHat = xCopy * ws
ax.plot(xCopy[:,1], yHat)
plt.show()

yHat = xMat * ws
    # vc=[1,2,39,0,8]
    # vb=[1,2,38,0,8]
    # print mean(multiply((vc-mean(vc)),(vb-mean(vb))))/(std(vb)*std(vc))
    # #corrcoef得到相关系数矩阵（向量的相似程度）
    # print corrcoef(vc,vb)
    # http://blog.csdn.net/u010156024/article/details/50419338
print(corrcoef(yHat.T, yMat)) # 计算2个矩阵的相似度