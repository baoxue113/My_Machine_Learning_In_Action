# coding=utf-8
# 从文本中构建词向量
import bayes
import numpy

# listOPosts:特征,listClasses:标签
listOPosts,listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
print myVocabList

print bayes.setOfWords2Vec(myVocabList,listOPosts[0]) # 这样好占空间

print bayes.setOfWords2Vec(myVocabList,listOPosts[3]) # 这样好占空间


