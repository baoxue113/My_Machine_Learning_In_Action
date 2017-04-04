# coding=utf-8
'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir

# inX : 要预测的数据点
# dataSet : 训练的数据
# labels : 训练数据的标签
# k : 取多少个最相似的点
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] # 获取数据矩阵有多少行
    diffMat = tile(inX, (dataSetSize,1)) - dataSet # 欧几里得相似度距离
    sqDiffMat = diffMat**2 # 平方乘
    sqDistances = sqDiffMat.sum(axis=1) # 求和
    distances = sqDistances**0.5 #开根号
    sortedDistIndicies = distances.argsort() #argsort：返回从小到大的索引
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # classCount.iteritems() : 排序的元素,key=operator.itemgetter(1) : 按元素的第二个属性排序，reverse=True ： 降序排列
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input('percentage of time spent plying video games(花费游戏时间百分比)?')) #用户输入参数
    ffMiles = float(raw_input('frequent flier miles earned per year?（每年获得的飞行常客里程数）'))
    iceCream = float(raw_input('liters of ice cream consumed per year?（每周消费的冰激凌公升数）'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr - minVals) / ranges,normMat,datingLabels,3)
    print "You will probably like this person: ",resultList[classifierResult - 1]


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    classLabelVectorIndex = set(classLabelVector) # 元素去重
    temp = []
    for value in classLabelVectorIndex: # set对象转数组
        temp.append(value)
    index = 0
    for value in classLabelVector: # 将字符标签转换成数字标识
        classLabelVector[index] = temp.index(value) + 1
        index += 1
    return returnMat,classLabelVector
    
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 获取矩阵每列的最小值
    maxVals = dataSet.max(0) # 获取矩阵每列的最大值
    ranges = maxVals - minVals # 每列的最大值减去最小值
    normDataSet = zeros(shape(dataSet)) # 制作和dataSet一样大小的矩阵
    m = dataSet.shape[0] # 获取矩阵有多少行
    temp = tile(minVals, (m,1))
    normDataSet = dataSet - tile(minVals, (m,1)) # 用矩阵的每个元素减去最小值
    temp1 = tile(ranges, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1)) # 元素数据归一化后的矩阵  #element wise divide
    return normDataSet, ranges, minVals
   
def datingClassTest():
    hoRatio = 0.50 # 训练数据基数     #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat) # 归一化数据
    m = normMat.shape[0] # 获取数据行
    numTestVecs = int(m*hoRatio) #
    errorCount = 0.0
    for i in range(numTestVecs): # 预测与训练数据，前500个数据用来测试，后500个数据用来训练
        # 这里用未归一化的数据运行了一下，错误次数增加
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3) # 不能固定建模,每次预测计算量巨大
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0 # 统计错误的次数
    print "the total error rate is: %f" % (errorCount/float(numTestVecs)) # 计算错误率
    print errorCount
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i] # 获取文件名
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # img2vector() ： 将文件矩阵转为数组向量
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3) #预测数据点
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))