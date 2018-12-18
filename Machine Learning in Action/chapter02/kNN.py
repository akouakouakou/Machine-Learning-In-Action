from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0( inX, dataSet, labels, k):
    """
    ----------------------------------------------------------------------------

    k-Nearest Neighbor算法

    输入：待分类的向量inX， 训练样本集dataSet， 训练样本的标签labels，最近邻数目k
    输出：向量inX的预测类别label

    参数
    inX：list，用于分类的输入向量    
    dataSet：array，输入的训练样本集    
    labels：list，训练样本集中每个样本的标签    
    k：int，最近邻居的数目

    示例
    >>> import kNN
    >>> group, labels = kNN.createDataSet()
    >>> group
    array([[1. , 1.1],
           [1. , 1. ],
           [0. , 0. ],
           [0. , 0.1]])
    >>> labels
    ['A', 'A', 'B', 'B']
    >>> kNN.classify0([0, 0], group, labels, 3)
    'B'
    
    ----------------------------------------------------------------------------
    """
    dataSetSize = dataSet.shape[0]    #获取dataSet行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet    #tile()将inX行数扩充为样本个数
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)   #矩阵内每个列表内的元素求和（计算两个向量点距离的平方）
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()    #获得距离从小到达的列表索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]    #最邻近的向量对应标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1    #统计最近邻重复的标签数量
    sortedClassCount = sorted(classCount.items(), \
                              key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]    #返回最近邻中出现频率最高的样本标签

def file2matrix(filename):
    """
    ----------------------------------------------------------------------------
    
    将文本记录转化为NumPy的解析程序

    输入：文件名filename
    输出：训练集矩阵returnMat，训练集样本标签classLabelVector

    参数
    filename：string，文件名字符串
    
    示例
    >>> datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
    >>> datingDataMat
    array([[4.0920000e+04, 8.3269760e+00, 9.5395200e-01],
           [1.4488000e+04, 7.1534690e+00, 1.6739040e+00],
           [2.6052000e+04, 1.4418710e+00, 8.0512400e-01],
           ...,
           [2.6575000e+04, 1.0650102e+01, 8.6662700e-01],
           [4.8111000e+04, 9.1345280e+00, 7.2804500e-01],
           [4.3757000e+04, 7.8826010e+00, 1.3324460e+00]])
    >>> datingLabels[0:20]
    [3, 2, 1, 1, 1, 1, 3, 3, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3]
    
    ----------------------------------------------------------------------------
    """
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    """
    ----------------------------------------------------------------------------

    归一化特征值

    输入：训练样本集dataSet
    输出：归一化的训练集矩阵normDataSet，最大值最小值值差ranges，最小值minVals

    参数
    dataSet：array，输入的训练样本集

    示例
    >>> normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
    >>> normMat
    array([[0.44832535, 0.39805139, 0.56233353],
           [0.15873259, 0.34195467, 0.98724416],
           [0.28542943, 0.06892523, 0.47449629],
           ...,
           [0.29115949, 0.50910294, 0.51079493],
           [0.52711097, 0.43665451, 0.4290048 ],
           [0.47940793, 0.3768091 , 0.78571804]])
    >>> ranges
    array([9.1273000e+04, 2.0919349e+01, 1.6943610e+00])
    >>> minVals
    array([0.      , 0.      , 0.001156])

    ----------------------------------------------------------------------------
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    """
    ----------------------------------------------------------------------------

    分类器针对约会网站测试代码

    输入：NULL
    输出：测试样本的预测类别classifierResult，测试样本的真实类别datingLabels[i]，错误率errorRate

    示例
    kNN.datingClassTest()
    the classifier came back with: 3, the real answer is: 3
    the classifier came back with: 2, the real answer is: 2
    .
    .
    the classifier came back with: 3, the real answer is: 3
    the classifier came back with: 3, the real answer is: 3
    the classifier came back with: 2, the real answer is: 2
    the classifier came back with: 1, the real answer is: 1
    the classifier came back with: 3, the real answer is: 1
    the total error rate is: 0.050000
    
    ----------------------------------------------------------------------------
    """
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                     datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d"\
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def classifyPerson():
    """
    ----------------------------------------------------------------------------

    约会网站预测函数

    输入：NULL
    输出：对一个人的好感程度

    示例
    >>> kNN.classifyPerson()
    percentage of time spent playing video games?10
    frequent flier miles earned per year?10000
    liters of ice cream consumed per year?0.5
    You will probably like this person: in small doses

    ----------------------------------------------------------------------------
    """
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:",resultList[classifierResult - 1])

def img2vector(filename):
    """
    ----------------------------------------------------------------------------

    将图像转换为测试向量

    输入：文件名filename
    输出：图像向量returnVect

    参数
    filename：string，文件名字符串

    示例
    >>> testVector = kNN.img2vector('digits/testDigits/0_13.txt')
    >>> testVector[0,0:31]
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> testVector[0,32:63]
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.,
           1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    ----------------------------------------------------------------------------
    """
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    """
    ----------------------------------------------------------------------------

    手写数字识别系统的测试代码

    输入：NULL
    输出：测试样本的预测类别classifierResult，测试样本的真实类别classNumStr，
          错误预测样本数errorCount，错误率errorRate

    示例
    >>> kNN.handwritingClassTest()
    the classifier came back with: 0, the real answer is: 0
    the classifier came back with: 0, the real answer is: 0
    .
    .
    the classifier came back with: 7, the real answer is: 7
    the classifier came back with: 7, the real answer is: 7
    the classifier came back with: 8, the real answer is: 8
    the classifier came back with: 8, the real answer is: 8
    the classifier came back with: 8, the real answer is: 8
    .
    .
    the classifier came back with: 9, the real answer is: 9
    the total number of errors is: 10
    the total error rate is: 0.010571

    ----------------------------------------------------------------------------
    """
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d"\
              % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
