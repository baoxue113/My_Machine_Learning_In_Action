# coding=utf-8
'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

# dataSet:要切分的数据, feature：要切分的特征（代表第几列的索引）, value：切分的数据的判断值
def binSplitDataSet(dataSet, feature, value):
    temp1 = dataSet[:,feature]
    temp2 = nonzero(dataSet[:,feature] > value)# 获取集合中每个元素大于目标值value的索引
    temp3 = nonzero(dataSet[:,feature] > value)[0]
    temp4 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]# 将符合条件的数据取出来，变成一个新的集合
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0,mat1

def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

def regErr(dataSet):
    temp2 = dataSet[:,-1] # 获取矩阵中的最后一个元素
    temp1 = var(dataSet[:, -1]) # 求向量的方差
    temp3 = shape(dataSet)[0]
    temp4 = var(dataSet[:,-1]) * shape(dataSet)[0] # 计算总方差
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

# 选择最好的切分特征索引和值 ops=(1,4) 1：是容许的误差下降值，4：是切分的最少样本数。
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    #是容许的误差下降值，误差下降不能少于这个数
    tolS = ops[0];
    #是切分的最少样本数。
    tolN = ops[1]
    #  if all the target variables are the same value: quit and return value
    # 如果所有的目标变量都是相同的值：退出和返回值
    temp1 = dataSet[:,-1] # 取矩阵的最后一列
    temp6 = dataSet[:, -1].T # 列转行
    temp2 = dataSet[:,-1].T.tolist() # 转成集合
    temp3 = dataSet[:,-1].T.tolist()[0] # 获取数组第0个元素
    temp4 = set(dataSet[:,-1].T.tolist()[0])# 去重和从小到大排序
    temp5 = len(set(dataSet[:,-1].T.tolist()[0])) # 获得元素个数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)# 计算平方误差
    bestS = inf; bestIndex = 0; # 最好的切分特征（列） bestValue = 0
    for featIndex in range(n-1): # n是列的总数
        temp7 = dataSet[:,featIndex] # 获取第featIndex列的所有值
        for splitVal in set(dataSet[:,featIndex]):# 用第featIndex列的每一个值尝试切分数据集合，然后计算切分后的2个集合的平方误差之和
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): # 如果切分后的2个集合元素个数都小于4，则进行下一次切分
                continue
            newS = errType(mat0) + errType(mat1)# 计算2个集合的平方误差和
            if newS < bestS: # 新误差比最好的误差小，更新数据
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: # S ：切分数据前的平方差 ，bestS:切分数据后的平方差 ，tolS：是容许的误差下降值，不能小于这个数
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue) # 用最优的列和值切分数据。
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #如果切分后的2个集合的行数少于指定的数，就不切集合 #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None:
        return val # 如果分裂击中停止条件返回 if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

#
def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree) # 如果我们没有测试数据崩溃树 if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):# 如果又边或者左边是一棵树，就切分测试数据集 如果树枝不是树，试着修剪它们 if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): # 如果左边是一棵树，就用切好的左边的数据来剪枝，递归
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): # 如果右边是一棵树，就用切好的右边的数据来剪枝，递归
        tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):# 如果左右都不是树
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal']) # 切分数据集
        temp1 = lSet[:,-1]
        temp2 = tree['left']
        temp3 = lSet[:,-1] - tree['left']
        temp4 = power(lSet[:,-1] - tree['left'],2)
        temp5 = sum(power(lSet[:,-1] - tree['left'],2))
        # 切分的测试数据左边向量 - 树左边的值，为了预防负数发生所以开平方
        temp6 = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2))
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0 # 树的左右值的平均数
        errorMerge = sum(power(testData[:,-1] - treeMean,2))# 测试数据集合的最优一列，减去平均数，开平方，求和
        if errorMerge < errorNoMerge: 
            #print "merging" # 合并树，取树的左右2个值得平均数
            return treeMean
        else:
            return tree
    else:
        return tree
    
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat