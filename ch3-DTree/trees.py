#encoding=utf-8
# 加上才能中文注释

from math import log  # 求熵要取对数
import operator


# 注：以下所有代码中label翻译为：
#    【标签】时表示特征的标签，即西瓜书上的【色泽，敲声...】
#    【标记】时表示输出y，即西瓜书上的【好瓜、坏瓜


def createDataSet():
    """
    这里的label(标签)不是西瓜书上的【好瓜、坏瓜】，即不是输出y(标记)。而是特征的标签，即【色泽，敲声...】
    :return: 数据集，特征标签
    """
    dataSet = [[1, 1, 'yes'],[1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']] # yes->鱼，no->非鱼
    labels = ['no surfacing', 'flippers']  # no surfacing->不浮出水面可以生存； flippers->有脚蹼
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    根据标签计算数据集的香农熵：信息的期望，Xi的信息 = -logp(Xi)，即熵 = -∑p(Xi)*logp(Xi)
    :param dataSet:
    :return:香农熵
    """
    # 求实例总数
    numEntries = len(dataSet)
    labelCounts = {}  # 创建字典记录各标记【标记->好瓜or坏瓜】出现的次数
    for featVec in dataSet:
        # 将当前实例的标记存储，即每一行数据的最后一个数据代表的是标记
        currentLabel = featVec[-1]
        # 为所有可能的标记创建字典，如果当前的键值不存在则将将当前键值加入字典，每个键值都记录了当前标记出现的次数。
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    #  根据各标记的占比，求香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries   # 根据频率计算概率
        shannonEnt -= prob * log(prob, 2)           # 计算香农熵，以 2 为底求对数
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集：将dataset[i][axis] = value 的实例选出来并删掉ataset[i][axis]这一列
    即依据axis列进行分类，如果axis列的值等于value的时候，就划分到我们创建的新的数据集中
    :param dataSet: 待划分数据集
    :param axis:    划分数据集的特征索引(第axis列)
    :param value:   目标特征的值
    :return:
    """
    retDataSet = []  # 为了不修改原数据集（因为函数参数为列表时python用引用传递）故创建一个新列表(其元素也是列表)
    for featVec in dataSet:
        if featVec[axis] == value: # 一旦发现符合要求的值则添加到新列表中
            reducedFeatVec = featVec[:axis]  # [:axis]表示前axis行，就是取featVec的前axis行

            # 关于extend和append的区别：
            # result = []
            # result.extend([1,2,3])  ->  result = [1, 2, 3]
            # result.append([4,5,6])  ->  result = [1, 2, 3, [4, 5, 6]]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    用不同特征划分数据集，计算熵，选择最好的特征
    :param dataSet:元素为列表的列表且每个实例最后一个元素是标记
    :return:划分后熵最小的特征(最好的)特征
    """
    numFeatures = len(dataSet[0]) - 1    # Feature数, 最后一列是标记列所以减一
    baseEntropy = calcShannonEnt(dataSet)# 数据集的原始信息熵
    bestInfoGain, bestFeature = 0.0, -1  # 最优的信息增益值, 和最优的Feature编号

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] # 获取第i个特征的的所有取值，会重复
        uniqueVals = set(featList)  # 使用set去重
        newEntropy = 0.0

        # 计算每种划分方式下的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet)) # 计算概率
            newEntropy += prob * calcShannonEnt(subDataSet)

        # gain[信息增益]: 熵的减少量 (数据无序度的减少)
        # 取所有特征中的信息增益最大的特征，即划分后熵越少越好【划分后越有序越好】
        infoGain = baseEntropy - newEntropy
        # print 'infoGain=', infoGain
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """
    创建决策树时，当用完了所有属性但类标记依然不是唯一的，此时调用此函数采用多数表决的方法决定叶子节点的分类
    :param classList: 已经没有属性了的标记列表
    :return: 出现次数最多的那个标记
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    递归创建树(字典形式)
    :param dataSet:数据集
    :param labels: 标签列表
    :return: 标记或树
    """
    classList = [example[-1] for example in dataSet]  # 数据集的所有标记
    # 第一个停止条件:所有的标记完全相同，则直接返回该标记
    if classList.count(classList[0]) == len(classList): # count(x) 统计x在list中出现的次数
        return classList[0]

    # 第二个停止条件:使用完了所有特征，仍然不能将数据集划分成仅包含唯一标记的分组，就选其中出现次数最多的那个标记
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的列(特征)
    bestFeat = chooseBestFeatureToSplit(dataSet) # 标签索引，如0，1
    bestFeatLabel = labels[bestFeat] # 标签具体值，如no surfacing，flippers

    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat]) # 删除这个标签
    featValues = [example[bestFeat] for example in dataSet]  # 获取bestFeat(最优标签)所有取值，会重复
    uniqueVals = set(featValues)  # 去重
    for value in uniqueVals:
        subLabels = labels[:] # 当函数参数是列表型时，python按引用传递。所以复制一遍
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # print 'myTree', value, myTree
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    使用决策树的分类函数
    :param inputTree:决策树模型
    :param featLabels:标签对应的名称
    :param testVec:测试输入的数据
    :return:分类的结果，即标记
    """
    firstStr = inputTree.keys()[0]   # 获取根节点的key值，即一个标签，如no surfacing
    secondDict = inputTree[firstStr] # 通过key得到根节点对应的value，即根节点的所有子树(仍为一个字典)
    featIndex = featLabels.index(firstStr) # 获得标签索引
    key = testVec[featIndex]  # 获得待测数据对应标签的具体标签值，如0或者1
    valueOfFeat = secondDict[key] # 根据具体的标签值进入到具体子树
    if isinstance(valueOfFeat, dict):   # 判断是否到达叶子: 判断valueOfFeat是否是dict类型
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree, filename):
    """
    数据很多时每次建树很分时间，故将建好的树存储到文件中，以后使用直接用grabTree(filename)调用
    """
    import pickle  # 序列化对象所需库
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()
def grabTree(filename):
    """
    加载文件
    """
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def lensesTest(filename):
    """
    使用决策树预测隐形眼镜类型并画出决策树
    """
    import treesPlot
    fr = open(filename)
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate'] # 标签
    lensesTree = createTree(lenses, lensesLabels)
    treesPlot.createPlot(lensesTree)


if __name__ == '__main__':
    myData,labels = createDataSet()
    print "数据集myData: ",myData
    print "特征labels: ",labels
    # print "\n香农熵 = ",calcShannonEnt(myData)

    # print '\n',splitDataSet(myData, 0, 1) # myData中第0列值等于1的数据
    # print splitDataSet(myData, 0, 0)

    # print chooseBestFeatureToSplit(myData)

    tmplabels = labels[:]  # 创建树时会改动labels，所以复制给tmplabels用它建树
    mytree = createTree(myData,tmplabels)
    print "决策树（字典形式）:",mytree

    print classify(mytree,labels,[1,0])

    storeTree(mytree,"tree_classifier.txt")
    print grabTree("tree_classifier.txt")

    lensesTest("lenses.txt")



