# coding=utf-8
'''
Created on Oct 19, 2010

@author: Peter
'''
# classifyNB(array(wordVector),p0V,p1V,pSpam): 预测数据
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

# 将文章单词去重制作成单词向量
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        temp1 = vocabSet
        temp2 = set(document)
        vocabSet = vocabSet | set(document) # 联合两个集合 代码：_test_.py#union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList) # 制作最长向量
    for word in inputSet: # inputSet：输入的文章类别的切好后的词
        if word in vocabList: # 如果词在vocabList总词中有
            returnVec[vocabList.index(word)] = 1 # 将returnVec中对于词的位置赋值未1，这里未考虑一个词出现2次的情况，如果考虑，应该是+=1吧
                                                 # +=1是词袋模型， = 1是词集模型
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)# 获知有多少个文档
    numWords = len(trainMatrix[0])# 获知所有文档共有多少个单词
    # 下一行是计算，全文章中，获取一篇文章是侮辱性文章的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs) # trainCategory，1代表侮辱性文章，0代表正常文章
    p0Num = ones(numWords); # 这里用到了特征平滑技术，随书62页有解释 # 制作32列的向量
    p1Num = ones(numWords)  # 这里用到了特征平滑技术，随书62页有解释 #change to ones()
    p0Denom = 2.0; # 统计类别为0，也就是正常文章出现的总单词数      随书62页有解释
    p1Denom = 2.0  # 统计类别为1，也就是侮辱性文章出现的总单词数    随书62页有解释                  #change to 2.0
    for i in range(numTrainDocs): # 循环遍历每一篇文章
        if trainCategory[i] == 1: # 如果此文章是侮辱性文章
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i]) # 属于侮辱性文章的单词一共有多少个
        else:
            temp1 = p0Num
            temp2 = trainMatrix[i]
            p0Num += trainMatrix[i] # 我猜这里用到了特征平滑技术,防止概率相除=0的情况 然后0又相乘，结果会是0
            temp5 = p0Num
            temp3 = p0Denom
            temp4 = sum(trainMatrix[i]) # 统计该文章类别中出现多少个单词
            p0Denom += sum(trainMatrix[i]) # 属于正常文章的单词一共有多少个
    # 下一行是计算每个特征属于侮辱性文章的概率
    p1Vect = log(p1Num/p1Denom) # 这里加log是：我们知道，当特征很多的时候，大量小数值的小数乘法会有溢出风险。因此，通常的实现都是将其转换为log：
    # 下一行是计算每个特征属于正常文章的概率
    p0Vect = log(p0Num/p0Denom) # http://blog.csdn.net/lsldd/article/details/41542107         #change to log()
    return p0Vect,p1Vect,pAbusive # p0Vect：每个特征属于正常文章的概率,p1Vect：每个特征属于侮辱性文章的概率,pAbusive:一篇文章是侮辱性文章的概率

# vec2Classify：预测的数据
# p0V:每个特征属于正常文章的概率
# p1V：每个特征属于侮辱性文章的概率
# pClass1：一篇文章是侮辱性文章的概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    temp1 = vec2Classify * p1Vec # 每个特征属于侮辱性文章的概率
    temp2 = log(pClass1)
    temp3 = sum(vec2Classify * p1Vec) # sum(vec2Classify * p1Vec)：预测特征是侮辱性文章的概率总和
    p1 = sum(vec2Classify * p1Vec) + log(pClass1) # log(pClass1)：一篇文章是侮辱性文章的概率   #element-wise mult
    temp4 = vec2Classify * p0Vec # 这点理解不了
    temp5 = log(1.0 - pClass1) # 一篇文章是正常文章的概率
    temp6 = sum(vec2Classify * p0Vec) # 预测每个特征是正常文章的概率总和
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1) # log(1.0 - pClass1)：一篇文章是正常文章的概率
    if p1 > p0:
        return 1
    else: 
        return 0
    
def bagOfWords2VecMN(vocabList, inputSet):# 词袋模型处理数据
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 # 词袋模型
    return returnVec

def testingNB():
    listOPosts,listClasses = loadDataSet() # listOPosts:特征，listClasses:标量
    myVocabList = createVocabList(listOPosts) # 创建所有文章中，所有单词的向量，有去重效果，每个单词是唯一的
    trainMat=[]
    for postinDoc in listOPosts: # 处理下数据
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses)) # p0V:每个特征属于正常文章的概率，p1V：每个特征属于侮辱性文章的概率，pAb：一篇文章是侮辱性文章的概率
    testEntry = ['love', 'my', 'dalmation'] # 创建测试数据
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry)) # 将测试数据制作成[1,0,0,0,1......]的向量
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)) # 预测数据点
    testEntry = ['stupid', 'garbage'] # 创建测试数据
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry)) # 将测试数据制作成[1,0,0,0,1......]的向量
    print (testEntry,('classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))) # 预测数据点

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString) #去掉.号，并且切词
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 只要单词长度大于2的单词
    
def spamTest():
    docList=[];
    classList = [];
    fullText =[] # 所有文章的单词集合，不带去重
    for i in range(1,26):
        temp1 = 'email/spam/%d.txt' % i
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList) # 二维数组
        fullText.extend(wordList) # http://www.cnblogs.com/awpboxer/p/5197282.html
        classList.append(1) # 辱骂的文章
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList) # 二维数组
        fullText.extend(wordList)
        classList.append(0) # 正常的文章
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = range(50); testSet=[]           #create test set
    for i in range(10): # 随机取10篇文章，用来做测试
        randIndex = int(random.uniform(0,len(trainingSet))) # 随机产生0 到 len(trainingSet)
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; # 训练数据矩阵
    trainClasses = [] # 训练类别向量
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex])) # 用词袋模型处理数据
        trainClasses.append(classList[docIndex])
    # p0V：每个特征属于正常文章的概率,
    # p1V：每个特征属于侮辱性文章的概率,
    # pSpam: 一篇文章是侮辱性文章的概率
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses)) # 计算特征概率
    errorCount = 0 # 分类错误统计
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex]) # 用词袋模型将数据制作成词向量
        # classifyNB(array(wordVector),p0V,p1V,pSpam): 预测数据
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]: # 判断预测类别是否分类错误
            errorCount += 1 # 如果分错，错误次数+1
            print ("classification error",docList[docIndex])
    print ('the error rate is: ',float(errorCount)/len(testSet)) # 计算错误率
    #return vocabList,fullText

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print (item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print (item[0])
