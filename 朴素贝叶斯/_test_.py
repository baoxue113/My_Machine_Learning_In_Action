# coding=utf-8
# 从文本中构建词向量
import bayes
import numpy
temp1 = set([1,2,3,4,5])
temp2 = set([6])
print temp1 | temp2 # set([1, 2, 3, 4, 5, 6])
temp1 = set([1,2,3,4,5])
temp2 = set([1,2,3])
print temp1 | temp2 # set([1, 2, 3, 4, 5])


