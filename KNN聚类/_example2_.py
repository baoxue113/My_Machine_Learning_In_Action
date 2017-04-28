# coding=utf-8
import kNN
# group : 数据特征，labels : 数据标签
group,labels = kNN.createDataSet()
#
temp = kNN.classify0([0,0],group,labels,3)
print temp

