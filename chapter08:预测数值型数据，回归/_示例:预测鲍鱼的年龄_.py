import regression
from numpy import *
import matplotlib.pyplot as plt
abX, abY = regression.loadDataSet('abalone.txt')

print("预测0:99的数据，训练0:99的数据")
yHat01 = regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
yHat1 = regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
yHat10 = regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
print(regression.rssError(abY[0:99],yHat01.T))
print(regression.rssError(abY[0:99],yHat1.T))
print(regression.rssError(abY[0:99],yHat10.T))
print()
print("预测100:199的数据，训练0:99的数据")
yHat01 = regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
yHat1 = regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
yHat10 = regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
print(regression.rssError(abY[100:199],yHat01.T))
print(regression.rssError(abY[100:199],yHat1.T))
print(regression.rssError(abY[100:199],yHat10.T))

print("用简单的线性回归预测100:199的数据，训练0:99的数据")
# 与简单的线性回归做比较
ws = regression.standRegres(abX[0:99],abY[0:99])
yHat = mat(abX[100:199] * ws) # 预测100:199,的值
print()
print(regression.rssError(abY[100:199],yHat.T.A))
