#encoding=utf-8
# 加上才能中文注释

from numpy import *


"""
注：
在利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积以获得文档属于某个类别的概率，
即计算 p(w0|C1) * p(w1|C1) * p(w2|C1)...。如果其中一个概率值 为 0，那么最后的乘积也为 0。
为降低这种影响，可以将所有词的出现数初始化为 1，并将分母初始化为 2 (取1 或 2 的目的主要是为了保证分子和分母不为0)。
另一个遇到的问题是下溢出，这是由于太多很小的数相乘造成的。当计算乘积 p(w0|ci) * p(w1|ci) * p(w2|ci)... p(wn|ci) 时，
由于大部分因子都非常小，所以程序会下溢出或者得到不正确的答案。一种解决办法是对乘积取自然对数。
在代数中有 ln(a * b) = ln(a) + ln(b), 于是通过求对数可以避免下溢出或者浮点数舍入导致的错误。
"""


def loadDataSet():
    """
    已经知道标记的文档和各文档标记(侮辱性文档or不侮辱)
    :return:postingList是进行切分后的文档列表，classVec是标记
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 文档0
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], # 文档1
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], # 文档2
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],  # 文档3
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], # 文档4
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']] # 文档5
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性
    return postingList, classVec

def createVocabList(dataSet):
    """
    创建一个所有文档中出现过的词汇表，用set去重
    """
    vocabSet = set([]) # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 操作符 | 用于求两个集合的并集（同按位或的符号）
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
    不计某个词出现的次数，只记录是否出现，即【词集模型set-of-words model】
    :param vocabList:词汇表
    :param inputSet:某个经过处理的文档[只包含单词]
    :return:文档向量，向量的元素为0或1，分别表示词汇表中的词汇是否在该文档出现
    """
    returnVec = [0] * len(vocabList) # 创建一个和词汇表等长的全0向量
    for word in inputSet:  # 遍历文档中的所有单词
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1 # 如果出现，则将输出的文档向量中的对应值设为1
        else:
            print "Error: the word %s is not in my Vocabulary!" % word
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    """
    计录某个词出现的次数，即【词袋模型bag-of-words model】
    :param vocaList:词汇表
    :param inputSet: 某个经过处理的文档[只包含单词]
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in inputSet:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数：p(Ci|w) = [p(w|Ci)*p(Ci)] / p(w) ,其中C1或C0表示是否是侮辱性文档
    :param trainMatrix: 各文档单词矩阵 [[1,0,1,1,1....],[],[]...]
    :param trainCategory: 各文档对应的标记[0,1,1,0....]，其中的1代表对应文档是侮辱性，0代表非侮辱性
    :return:p0Vect(取对数后的非侮辱性文档下各词出现的概率p(wi|C0)列表), p1Vect, pAbusive(侮辱性文档出现概率p(C1))
    """
    numTrainDocs = len(trainMatrix)  # 文档
    numWords = len(trainMatrix[0])  # 单词数
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 侮辱性文档的出现概率p(C1)，即文档标记中1出现的频率

    # 计算p(w0|Ci)*p(w1|Ci)*p(w2|Ci)...时,其中一个值为0时最终结果也为0
    # 为了消除这种影响，将所有词出现次数初始化为1，单词总数初始化为2
    p0Num = ones(numWords)  # 构造单词出现次数列表
    p1Num = ones(numWords)
    p0Denom = 2.0 # 整个数据集单词出现总数
    p1Denom = 2.0
    for i in range(numTrainDocs):   # 遍历所有文档
        if trainCategory[i] == 1:   # 是侮辱性文档
            # 对侮辱性文档的向量进行加和，[0,1,1,....] + [0,1,1,....]->[0,2,2,...]
            p1Num += trainMatrix[i]    #  累加每个词出现的词数
            p1Denom += sum(trainMatrix[i]) # 文档总词数累加
        else:  # 不是侮辱性文档
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 类别1,侮辱性
    # p1Vect = p1Num / p1Denom  #每个单词出现的概率：[1,2...]/90->[1/90,...]，即侮辱性文档的[P(w1|C1),P(w2|C1)....]
    # 当多个很小的数相乘时可能会出现下溢出即结果为0，所以这里将概率修正为取对数后的概率，以后的概率相乘就为概率对数相加
    p1Vect = log(p1Num / p1Denom)
    # 类别0
    # p0Vect = p0Num / p0Denom
    p0Vect = log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类函数
    :param vec2Classify:待测数据如[0,1,1,1,1...]，即要分类的词汇向量
    :param p0Vec:标记0即非侮辱性文档中各词出现的概率(修正并取了对数)列表
    :param p1Vec:标记1即侮辱性文档中各词出现的概率(修正并取了对数)列表
    :param pClass1:标记1出现的概率
    :return: 标记 0 or 1
    """
    # p(Ci|w) = [p(w|Ci)*p(Ci)] / p(w)，分子取了对数变成了加法sum
    # 下面的计算没有考虑分母P(w)，P(w)指的是此文档在所有的文档中出现的概率，所以 P(w) 是相同的。
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # 贝叶斯准则的分子
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    """
    贝叶斯分类器测试函数封装
    """
    # 1. 加载数据集
    listOPosts, listClasses = loadDataSet()
    # 2. 创建单词集合
    myVocabList = createVocabList(listOPosts)
    # 3. 计算单词是否出现并创建数据矩阵
    trainMat = []
    for postinDoc in listOPosts:
        # 返回m*len(myVocabList)的矩阵， 记录的都是0，1信息
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 4. 训练数据
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    # 5. 测试数据
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)


# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓实例：使用朴素贝叶斯过滤垃圾邮件↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
"""
收集数据: 提供文本文件
准备数据: 将文本文件解析成词条向量
分析数据: 检查词条确保解析的正确性
训练算法: 使用我们之前建立的 trainNB() 函数
测试算法: 使用朴素贝叶斯进行交叉验证
使用算法: 构建一个完整的程序对一组文档进行分类，将错分的文档输出到屏幕上
"""
def textParse(bigString):
    """
    将一个很大的字符串解析为小字符串列表（单词列表），并将所有字母转换成小写，再过滤掉不足三个字母的字符串
    :param bigString: 大字符串文本
    :return: 小字符串列表（单词列表）
    """
    import re # 使用正则表达式来切分句子，其中分隔符是除单词、数字外的任意字符串
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] # lower将所有字母换成小写

def spamTest():
    """
    垃圾邮件分类器自动化处理
    :return:对测试集中的50（25+25）封邮件进行分类，若邮件分类错误，则错误数加 1，最后返回总的错误百分比。
    """
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 切分，解析数据，并归类为 1 类别
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # 切分，解析数据，并归类为 0 类别
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList) # 创建词汇表
    trainingSet = range(50)
    testSet = []
    for i in range(10):# 随机取 10 个邮件用来测试
        randIndex = int(random.uniform(0, len(trainingSet))) #random.uniform(x, y)随机生成一个范围为x~y的实数
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    # 训练
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))

    # 测试
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
    if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
        errorCount += 1
    print 'the errorCount is: ', errorCount
    print 'the testSet length is :', len(testSet)
    print 'the error rate is :', float(errorCount) / len(testSet)
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑实例：使用朴素贝叶斯过滤垃圾邮件↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print "词汇表：",myVocabList
    print "将第0个文档转成词汇向量：",setOfWords2Vec(myVocabList, listOPosts[0])

    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(trainMat, listClasses)
    print "\n侮辱性文档占比p(C1):",pAb
    print "非侮辱文档中词汇表中各词出现概率p(wi|C0)【修正后】：\n",p0V
    print "侮辱文档中词汇表中各词出现概率p(wi|C1)【修正后】：\n", p1V

    print "\n侮辱性文档测试："
    testingNB() # 侮辱性文档测试

    print "\n邮件分类系统测试："
    spamTest()  # 邮件分类系统测试

