from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))
    '''
    if all(inX) >= 0:
        return 1.0/(1+exp(-inX))
    else:
        return exp(inX)/(1+exp(inX))
    '''

def gradAscent(dataMatIn, classLabels):
    """
    ----------------------------------------------------------------------------

    Logistic Regression梯度上升优化算法

    输入：训练样本集dataMatIn， 训练样本的标签classLabels
    输出：参数向量weights

    参数
    dataMatIn：list[list]，用于分类的输入向量集   
    classLabels：list，训练样本集中每个样本的标签
    
    示例
    >>> import logRegres
    >>> dataArr, labelMat = logRegres.loadDataSet()
    >>> logRegres.gradAscent(dataArr,labelMat)
    matrix([[ 4.12414349],
            [ 0.48007329],
            [-0.6168482 ]])
    
    ----------------------------------------------------------------------------
    """
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def plotBestFit(weights):
    """
    ----------------------------------------------------------------------------

    绘制数据集和Logistic回归最佳拟合直线的函数

    输入：参数向量weights
    输出：拟合后的函数图

    参数
    weights：matrix，参数向量
    
    示例
    >>> weights = logRegres.gradAscent(dataArr,labelMat)
    >>> logRegres.plotBestFit(weights.getA())
    
    ----------------------------------------------------------------------------
    """
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    """
    ----------------------------------------------------------------------------

    随机梯度上升算法

    输入：训练样本集dataMatrix， 训练样本的标签classLabels
    输出：参数向量weights

    参数
    dataMatrix：list[list]，用于分类的输入向量集   
    classLabels：list，训练样本集中每个样本的标签
    
    示例
    >>> dataArr, labelMat = logRegres.loadDataSet()
    >>> weights = logRegres.stocGradAscent0(array(dataArr),labelMat)
    >>> logRegres.plotBestFit(weights)
    
    ----------------------------------------------------------------------------
    """
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    ----------------------------------------------------------------------------

    改进的随机梯度上升算法

    输入：训练样本集dataMatrix，训练样本的标签classLabels，迭代次数numIter
    输出：参数向量weights

    参数
    dataMatrix：list[list]，用于分类的输入向量集   
    classLabels：list，训练样本集中每个样本的标签
    numIter：int，迭代次数
    
    示例
    >>> dataArr, labelMat = logRegres.loadDataSet()
    >>> weights = logRegres.stocGradAscent1(array(dataArr),labelMat)
    >>> logRegres.plotBestFit(weights)
    
    ----------------------------------------------------------------------------
    """
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    """
    ----------------------------------------------------------------------------

    打开测试集和训练集，并对数据进行格式化处理的函数

    输入：NULL
    输出：分类错误率
   
    ----------------------------------------------------------------------------
    """
    frTrain = open('data/horseColicTraining.txt')
    frTest = open('data/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainingWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainingWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of the average test is : %f" % errorRate)
    return errorRate

def multiTest():
    """
    ----------------------------------------------------------------------------

    调用colicTest()10次并求结果的平均值

    输入：NULL
    输出：分类错误率的平均值
    
    示例
    >>> logRegres.multiTest()
    the error rate of the average test is : 0.462687
    the error rate of the average test is : 0.358209
    the error rate of the average test is : 0.373134
    the error rate of the average test is : 0.417910
    the error rate of the average test is : 0.343284
    the error rate of the average test is : 0.402985
    the error rate of the average test is : 0.358209
    the error rate of the average test is : 0.343284
    the error rate of the average test is : 0.373134
    the error rate of the average test is : 0.373134
    after 10 iteratinos the average error rate is: 0.380597
    
    ----------------------------------------------------------------------------
    """
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iteratinos the average error rate is: %f" \
          % (numTests, errorSum/float(numTests)))
