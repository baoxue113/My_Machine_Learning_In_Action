# coding=utf-8
from array import array

import kNN
import matplotlib
import matplotlib.pyplot as plt
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt') #数据预处理
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2]) #看不出什么区别
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2],datingLabels,datingLabels)
plt.show()
