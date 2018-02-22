#!/usr/bin/python
#encoding=utf-8

from numpy import *

def loadDataSet():
    '''
    加载数据集文件,返回两个列表
    '''
    dataList = []; labelList = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataList.append([1.0, float(lineArr[0]), float(lineArr[1])]) # 三维特征X0-X2,X0恒为1
        labelList.append(int(lineArr[2]))
    return dataList, labelList


def sigmoid(inX):
    '''
    sigmoid函数：1/(1+exp(-x))
    '''
    return 1.0/(1+exp(-inX))

def gradAscent(dataList, LabelList):
    '''
    梯度上升算法，缺点是每次迭代需要计算整个数据集，计算量大
    h(z) = 1/(1+exp(-z)),其中z = w0*x0 + ... + wn*xn, 以下代码中weights = [w0,w1,...,wn]的转置，即是列向量
    :param dataMatIn: 是一个2维NumPy矩阵，每列分别代表每个不同的特征，每行则代表每个训练样本
    :param classLabels:类别标记，它是一个 1*100 的行向量。为了便于矩阵计算，需要将该行向量转置为列向量，再赋给labelMat
    :return: [w0,w1,...,wn]的转置矩阵
    '''
    dataMat = mat(dataList)             # 转换为NumPy矩阵
    labelMat = mat(LabelList).transpose() # 转置为列向量并赋给labelMat
    m,n = shape(dataMat)  # m=数据量，n=特征数
    alpha = 0.001  # alpha代表向目标移动的步长
    maxCycles = 500  # 迭代次数
    weightsMat = ones((n,1))  # ones((n,1))创建一个长度和特征数相同的矩阵（不能为数组因为下面要进行矩阵乘法），其中的数全部都是1
    for k in range(maxCycles):
        h = sigmoid(dataMat * weightsMat)     # 矩阵乘法
        # print 'hhhhhhh====', h
        error = (labelMat - h)              # labelMat是实际值, 计算误差
        # 类似吴恩达英文讲义P5
        weightsMat = weightsMat + alpha * dataMat.transpose() * error
    return weightsMat


def stocGradAscent0(dataList, LabelsList):
    '''
    随机梯度上升算法
    梯度上升算法在每次更新回归系数时都需要遍历整个数据集，计算复杂度较高。
    一种改进方法是一次仅用一个样本点来更新回归系数，该方法称为随机梯度上升算法
    '''
    dataArr = array(dataList)
    m, n = shape(dataArr)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        # sum(dataMatrix[i]*weights)为了求z的值,z=w0*x0 + w1*x1 + w2*x2,此处求出的 h 是一个具体的数值，而不是一个矩阵
        h = sigmoid(sum(dataArr[i]*weights))
        error = LabelsList[i] - h
        # 吴恩达英文讲义P19随机梯度上升: 逻辑回归似然函数梯度 = x * (y - h), x为输入，y为实际标记，h为预测标记
        weights = weights + alpha * error * dataArr[i]
    return weights

def stocGradAscent1(dataList, LabelsList, numIter=150):
    '''
    随机梯度上升改进：
    1、第一处改进为 alpha 的值。alpha 在每次迭代的时候都会调整，这回缓解上面波动图的数据波动或者高频波动。另外，虽然 alpha 会随着迭代次数不断减
    少，但永远不会减小到 0，因为我们在计算公式中添加了一个常数项。
    2、第二处修改为 randIndex 更新，这里通过随机选取样本拉来更新回归系数。这种方法将减少周期性的波动。这种方法每次随机从列表中选出一个值，然后从
    列表中删掉该值(再进行下一次迭代)。
    3、增加了一个迭代次数作为函数参数，默认为150
        :return:
    '''
    dataArr = array(dataList)
    m,n = shape(dataArr)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001   # alpha 会随着迭代不断减小，但永远不会减小到0，因为后边还有一个常数项0.0001
            # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataArr[randIndex]*weights))
            error = LabelsList[randIndex] - h
            weights = weights + alpha * error * dataArr[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(dataList,labelList,weights):
    '''
    将我们得到的数据可视化展示出来
    '''
    import matplotlib.pyplot as plt
    weightsArr = array(weights)  # 转换为数组
    dataArr = array(dataList)
    # print type(dataArr),type(weightsArr)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelList[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s') # 散点图
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x1 = arange(-3.0, 3.0, 0.1)  # 横轴x1为-3到3间隔为0.1的系列点

    # sigmoid函数中z = 0为分界点，而 0 = z = w0*x0 + w1*x1 + w2*x2,其中x0 = 1，解出x2
    x2 = (-weightsArr[0]-weightsArr[1]*x1)/weightsArr[2]
    ax.plot(x1, x2)  # 画分割线
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 实例：从疝气病症预测病马的死亡率 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
def classifyVector(inX, weights):
    '''
    根据sigmoid函数值预测类别标记
    :return: 0 or 1
    '''
    inXArr = array(inX)
    prob = sigmoid(sum(inXArr * weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    '''
    打开测试集和训练集，并对数据进行格式化处理
    :return: errorRate -- 分类错误率
    '''
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        # print len(currLine)
        lineArr =[]
        for i in range(21): # len(currLine) = 22
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(trainingSet, trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(lineArr, trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

def multiTest(numTests = 5):
    '''
    多次调用并取平均值
    :return:
    '''
    errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))

# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 实例：从疝气病症预测病马的死亡率 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


if __name__ == '__main__':
    # dataList,labelList = loadDataSet()
    #
    # weights1 = gradAscent(dataList,labelList)
    # weights2 = stocGradAscent0(dataList,labelList)
    # weights3 = stocGradAscent1(dataList,labelList)
    # print type(weights1),type(weights2),type(weights3)
    #
    # plotBestFit(dataList, labelList, weights1)
    # plotBestFit(dataList, labelList, weights2)
    # plotBestFit(dataList, labelList, weights3)

    colicTest() # 会报错警告浮点数计算会溢出，暂时忽略这个报错
    multiTest()

