from math import log  # 用于计算熵的对数函数
import operator  # 用于排序操作
import numpy as np  # 用于数组操作
import openpyxl        # 导入openpyxl库，用于读取Excel文件


def createDataSet_local():
    """构建数据集"""
    dataSet = [['yes', 'single', 125, 'no'],
               ['no', 'married', 100, 'no'],
               ['no', 'single', 70, 'no'],
               ['yes', 'married', 120, 'no'],
               ['no', 'divorced', 95, 'yes'],
               ['no', 'married', 60, 'no'],
               ['yes', 'divorced', 220, 'no'],
               ['no', 'single', 85, 'yes'],
               ['no', 'married', 75, 'no'],
               ['no', 'single', 90, 'yes']]
    labels = ['do or donot have a house', 'marriage', 'income(k)']  # 特征标签
    return dataSet, labels


def createDataSet_excel(path):
    # 从Excel文件中读取数据并生成数据集和标签
    workbook = openpyxl.load_workbook(path)  # 打开Excel文件
    sheet = workbook.active  # 获取活动工作表
    dataSet = []  # 初始化数据集
    labels = [cell.value for cell in sheet[1][1:]]  # 获取特征标签（第一行的值，去掉第一列）

    for row in sheet.iter_rows(min_row=2):
        dataSet.append([cell.value for cell in row[1:]])  # 逐行读取数据并加入数据集

    return dataSet, labels  # 返回数据集和标签


def calcShannonEnt(dataSet):
    """
    计算给定数据集的香农熵
    :param dataSet: 给定的数据集
    :return: 返回香农熵
    """
    numEntries = len(dataSet)  # 数据集中的条目数
    labelCounts = {}  # 用于统计各类别的数量
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 获取当前条目的类别标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for label in labelCounts.keys():
        prob = float(labelCounts[label])/numEntries  # 计算每个类别的概率
        shannonEnt -= prob*log(prob, 2)  # 累加计算香农熵
    return shannonEnt


def majorityCnt(classList):
    """获取出现次数最多的分类名称"""
    classCount = {}
    classList = np.mat(classList).flatten().A.tolist()[0]  # 转换为一维列表
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def splitDataSet(dataSet, axis, value):
    """对离散型特征划分数据集"""
    retDataSet = []  # 返回的数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])  # 去除当前特征
            retDataSet.append(reducedFeatVec)
    return retDataSet


def splitDataSet(dataSet, axis, value):
    """对离散型特征划分数据集"""
    retDataSet = []  # 返回的数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])  # 去除当前特征
            retDataSet.append(reducedFeatVec)
    return retDataSet


def splitContinuousDataSet(dataSet, axis, value, direction):
    """对连续型特征划分数据集"""
    subDataSet = []
    for featVec in dataSet:
        if direction == 0:  # 大于指定值
            if featVec[axis] > value:
                reduceData = featVec[:axis]
                reduceData.extend(featVec[axis + 1:])
                subDataSet.append(reduceData)
        if direction == 1:  # 小于等于指定值
            if featVec[axis] <= value:
                reduceData = featVec[:axis]
                reduceData.extend(featVec[axis + 1:])
                subDataSet.append(reduceData)
    return subDataSet


def chooseBestFeatureToSplit(dataSet, labels):
    """选择最好的数据集划分方式"""
    baseEntropy = calcShannonEnt(dataSet)  # 基础熵值
    baseGainRatio = 0.0  # 初始化增益率
    bestFeature = -1  # 初始化最优特征索引
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    bestSplitDic = {}  # 存储连续型特征的最佳切分点

    for i in range(numFeatures):
        featVals = [example[i] for example in dataSet]  # 获取第i个特征的值
        if type(featVals[0]).__name__ == 'float' or type(featVals[0]).__name__ == 'int':
            sortedFeatVals = sorted(featVals)
            splitList = [(sortedFeatVals[j] + sortedFeatVals[j + 1]
                          ) / 2.0 for j in range(len(featVals) - 1)]
            for value in splitList:
                newEntropy = 0.0
                greaterSubDataSet = splitContinuousDataSet(
                    dataSet, i, value, 0)
                smallSubDataSet = splitContinuousDataSet(dataSet, i, value, 1)
                prob0 = len(greaterSubDataSet) / float(len(dataSet))
                newEntropy += prob0 * calcShannonEnt(greaterSubDataSet)
                prob1 = len(smallSubDataSet) / float(len(dataSet))
                newEntropy += prob1 * calcShannonEnt(smallSubDataSet)
                splitInfo = -prob0 * log(prob0, 2) - prob1 * log(prob1, 2)
                gainRatio = float(baseEntropy - newEntropy) / splitInfo
                if gainRatio > baseGainRatio:
                    baseGainRatio = gainRatio
                    bestSplit = value
                    bestFeature = i
            bestSplitDic[labels[i]] = bestSplit
        else:
            uniqueVals = set(featVals)
            newEntropy = 0.0
            splitInfo = 0.0
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                splitInfo -= prob * log(prob, 2)
                newEntropy += prob * calcShannonEnt(subDataSet)
            if splitInfo == 0.0:
                continue
            gainRatio = float(baseEntropy - newEntropy) / splitInfo
            if gainRatio > baseGainRatio:
                bestFeature = i
                baseGainRatio = gainRatio

    if type(dataSet[0][bestFeature]).__name__ in ['float', 'int']:
        bestFeatValue = bestSplitDic[labels[bestFeature]]
    else:
        bestFeatValue = labels[bestFeature]
    return bestFeature, bestFeatValue


def createTree(dataSet, labels):
    """创建C4.5树"""
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 所有类别相同，停止划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 所有特征已用完，返回多数类别
        return majorityCnt(classList)
    bestFeature, bestFeatValue = chooseBestFeatureToSplit(dataSet, labels)
    if bestFeature == -1:
        return majorityCnt(classList)
    bestFeatLabel = labels[bestFeature]
    myTree = {bestFeatLabel: {}}
    subLabels = labels[:bestFeature] + labels[bestFeature + 1:]
    if type(dataSet[0][bestFeature]).__name__ == 'str':
        featVals = [example[bestFeature] for example in dataSet]
        uniqueVals = set(featVals)
        for value in uniqueVals:
            reduceDataSet = splitDataSet(dataSet, bestFeature, value)
            myTree[bestFeatLabel][value] = createTree(reduceDataSet, subLabels)
    else:
        value = bestFeatValue
        greaterSubDataSet = splitContinuousDataSet(
            dataSet, bestFeature, value, 0)
        smallSubDataSet = splitContinuousDataSet(
            dataSet, bestFeature, value, 1)
        myTree[bestFeatLabel]['>' +
                              str(value)] = createTree(greaterSubDataSet, subLabels)
        myTree[bestFeatLabel]['<=' +
                              str(value)] = createTree(smallSubDataSet, subLabels)
    return myTree


if __name__ == '__main__':
    path = 'data.xlsx'  # Excel文件路径
    # dataSet, labels = createDataSet_local()
    dataSet, labels = createDataSet_excel(path=path)
    mytree = createTree(dataSet, labels)
    print("最终构建的C4.5分类树为：\n", mytree)
