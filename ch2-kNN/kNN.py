#encoding=utf-8
# 加上才能中文注释

from numpy import *    #导出所有
import operator        #k近邻算法排序时要用
from os import listdir #手写识别函数中要用

# 示例
def createDataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # size为训练集行数(样本数)

    # 距离计算：差的平方和再开根号
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  # 计算差，tile为构造一个数组
    sqdiffMat = diffMat ** 2
    sqDistances = sqdiffMat.sum(axis = 1)
    #sum(axis=x)即第x维就消失，[[1,2,1],[3,4,3]]对0维求和:[4,6,4],对1维求和:[4,10]
    distances = sqDistances ** 0.5
    sortedDistances = distances.argsort()  #返回的是从小到大的下标，如[1.2,2.4,0]->[2,0,1]

    # 选择距离最小的k个点
    classCount = {} # 字典
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #若voteIlabel在字典中则value+1,否则设value为0再+1

    # 返回前k个点出现频率最高的类别
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse=True) # 从大到小，逆序
    return sortedClassCount[0][0] # 返回value最大对应的key


def file2matrix(filename):
    '''
    将datingTestSet2.txt特征数据文件转换成训练样本矩阵和类标签向量
    '''
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3)) # zeros((2,3))就是生成一个 2*3的矩阵，各个位置上全是 0
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 去掉首尾的空格
        listFromLine = line.split('\t') # 返回列表
        returnMat[index, :] = listFromLine[0:3] # 每列的属性数据
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    '''
    归一化特征值
    :param dataSet
    :return: 归一化后的dataSet
    归一化公式:
    Y = (X-Xmin)/(Xmax-Xmin)
    其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    '''
    # 计算每种属性的最大值、最小值、范围
    minVals = dataSet.min(0)  # minVals中有多个值，即每种属性最小值,axis = 0
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 将最小值之差除以范围组成矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet ,ranges, minVals


def datingClassTest():
    """
    Desc:       对约会网站分类器的测试
    parameters: none
    return:     打印错误率
    """
    # 设置测试数据的的一个比例
    hoRatio = 0.1  # 10%作为测试剩下90%作为样本
    # 从文件中加载数据
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') # load data setfrom file # 归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    print 'numTestVecs=', numTestVecs
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %.2f%%" % (100*errorCount / float(numTestVecs))
    print errorCount


def classifyPerson():
    """
    Desc:       约会网站预测函数
    parameters: none
    return:     打印预测结果
    """
    resultList = ['不喜欢的人', '魅力一般的人', '极具魅力的人']
    ffMiles = float(raw_input("请输入每年飞行的里程数(如12000): "))
    percentTats = float(raw_input("请输入游戏时间所耗费百分比(如20): "))
    iceCream = float(raw_input("请输入每周消费的冰淇淋公升数(如0.8): "))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels, 3)
    print "经过分析, 这个人可能是: ", resultList[classifierResult - 1]

############################33####以下是手写数字识别相关函数##############################
def img2vector(filename):
    '''
    将一个样本即32X32的二进制图像转换成1X1024的向量，这样就可以使用上面的k-邻近分类器
    :param:  filename
    :return: 1X1024的向量
    '''
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    '''
    手写数字识别系统测试代码
    :param: none
    :return: 打印错误率
    '''
    # 1. 训练数据
    print "正在训练，请等待。。。\n"
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)  # m个样本
    trainingMat = zeros((m, 1024))
    # hwLabels存储0~9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]      # 去掉".txt"
        classNumStr = int(fileStr.split('_')[0]) # 获得值
        hwLabels.append(classNumStr)
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)

    # 2. 测试数据
    print "正在测试，请等待。。。\n"
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 2)
        # print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "错误数: %d" % errorCount
    print "错误率: %.2f%%" % (100*errorCount / float(mTest))
#########################################以上是手写数字识别相关函数##################


if __name__ == '__main__':
    group,labels = createDataset()
    #print classify0([1,1],group,labels,3)

    # dataMat, LabelVector = file2matrix("datingTestSet2.txt")
    # normDataMat = autoNorm(dataMat)
    # datingClassTest()
    #classifyPerson()
    handwritingClassTest()

    #print len(normDataMat)
    #print group.shape[0]
    # print labels