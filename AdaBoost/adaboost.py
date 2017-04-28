'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
#预测数据点
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1)) #构建向量，行数根据训练数据的行数来定，默认值为1
    if threshIneq == 'lt':#判断是左节点，还是右节点
        temp1= dataMatrix[:,dimen] <= threshVal #返回相应的ture和false
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0 #将数据中第dimen列的所有值中，<= threshVal的值，全部改为-1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    
# 生成多个单层决策树，也就是多个弱分类器
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr);
    labelMat = mat(classLabels).T #.T 转换矩阵
    m,n = shape(dataMatrix)
    numSteps = 10.0;
    bestStump = {};#表示最好的树桩
    bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    for i in range(n):  #根据特征列来预测数据 #loop over all dimensions
        temp1 = dataMatrix[:,i] #取所有行，第几列的数据

        rangeMin = dataMatrix[:,i].min(); # 取第i列的最小值
        rangeMax = dataMatrix[:,i].max(); # 取第i列的最大值
        stepSize = (rangeMax-rangeMin)/numSteps #移动的步数的大小
        for j in range(-1,int(numSteps)+1):#采用爬山法，来寻找最好的阀值（切分值） #loop over all range in current dimension
            for inequal in ['lt', 'gt']: #先看left,再看right #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize) # j越来越大，移动的步数也就越来越大,爬山法
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal) # 预测数据类型 #call stump classify with i, j, lessThan
                errArr = mat(ones((m,1))) #预测错误的数组
                errArr[predictedVals == labelMat] = 0 #如果预测的类型和真实的类型一致，将其值改为0，不做接下来的sum计算
                temp1 = D.T
                weightedError = D.T*errArr #根据分类的对错计算相应的错误权重，相当于将分对的次数，乘以默认值0.2 #calc total error multiplied by D
                print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError: # 如果错误权重，比以前的小
                    minError = weightedError
                    bestClasEst = predictedVals.copy() # python按对象引用传值
                    bestStump['dim'] = i #最好的特征列
                    bestStump['thresh'] = threshVal #阀值
                    bestStump['ineq'] = inequal#最好的左或右节点
    #bestStump：最好的决策树桩，minError：最小的错误权重，bestClasEst：最好的预测结果
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        #print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        #print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print ("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print (aggClassEst)
    return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print ("the Area Under the Curve is: ",ySum*xStep)
