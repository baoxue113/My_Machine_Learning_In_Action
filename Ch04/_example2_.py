# coding=utf-8
# 训练算法：从词向量计算概率
import bayes
import numpy

# listOPosts:特征,listClasses:标签
listOPosts,listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)

trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
print trainMat

#p0V:每个特征属于正常文章的概率，p1V：每个特征属于侮辱性文章的概率，pAb：一篇文章是侮辱性文章的概率
p0V,p1V,pAb = bayes.trainNB0(trainMat,listClasses)
print
print pAb



