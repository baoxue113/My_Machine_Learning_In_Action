import regression
from numpy import *
import matplotlib.pyplot as plt
xArr, yArr = regression.loadDataSet('abalone.txt')
temp1 = regression.stageWise(xArr, yArr, 0.01, 200)# 计算最佳回归系数
temp2 = regression.stageWise(xArr, yArr, 0.001, 5000)# 计算最佳回归系数

xMat = mat(xArr)
yMat = mat(yArr).T
xMat = regression.regularize(xMat) # 标准化数据
yM = mean(yMat, 0)
yMat = yMat - yM
weights = regression.standRegres(xMat, yMat.T)
print(weights.T)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(weights)
plt.show()


