# coding=utf-8
'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

# 函数的作用是：计算给定数据集的香浓信息熵的增益，香浓信息熵越大，划分的分类越多，结果越好
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) # 获取集合有多少元素
    labelCounts = {}
    for featVec in dataSet: #这个for循环的作用是统计不同标签的类别分别出现了多少次 #the the number of unique elements and their occurance
        currentLabel = featVec[-1] # 获取当前标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries # 计算每个标签出现的概率
        shannonEnt -= prob * log(prob,2) # 随书35页的第二个公式
    return shannonEnt

# myDat:待划分的数据集，第二个参数：代表选择第几列（也就是特征），第三个参数：用来划分的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     # 用于劈裂的轴 chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:]) # extend:书上38页有解说
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的特征来切分数据集
def chooseBestFeatureToSplit(dataSet):
    temp1 = len(dataSet[0])
    numFeatures = len(dataSet[0]) - 1 # 计算数据有多少特征
    baseEntropy = calcShannonEnt(dataSet)  # 计算原始集合的信息熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):  # 用第一列特征划分试试，用第二列特征划分试试......      #iterate over all the features
        featList = [example[i] for example in dataSet] # 获取第一列特征的所有值 #create a list of all the examples of this feature
        uniqueVals = set(featList) # 将值去重      #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)# 用第i个特征的第一种value的值计算信息熵，分别计算......
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 将计算出来的信息熵分别增加起来
        infoGain = baseEntropy - newEntropy # 用最好的信息熵减去新算出来的信息增益     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # 如果结果比最好的信息熵还大就更新值     #compare this to the best gain so far
            bestInfoGain = infoGain # 最好的信息熵赋值        #if better than current best, set to best
            bestFeature = i # 最好的特征赋值
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] # 获取每个数据点的标签
    temp1 = classList.count(classList[0]) # 计算有多少种标签
    temp3 = classList[0]
    temp2 = len(classList)
    if classList.count(classList[0]) == len(classList): # 当所有类都相等时停止拆分
        return classList[0]# 当所有类都相等时停止拆分 stop splitting when all of the classes are equal
    temp4 = len(dataSet[0])
    if len(dataSet[0]) == 1: # 当数据集中没有更多功能时停止拆分 stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) # 选择最好的特征来切分数据集
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]# 获取最好的特征值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        # splitDataSet() : # myDat:待划分的数据集，第二个参数：代表选择第几列（也就是特征），第三个参数：用来划分的值
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)# 递归建树
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
