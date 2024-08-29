import openpyxl        # 导入openpyxl库，用于读取Excel文件
from math import log   # 导入log函数，用于计算对数
import operator        # 导入operator模块，用于排序操作


def calcShannonEnt(dataSet):
    # 计算数据集的香农熵（Shannon Entropy），用于衡量数据集的混乱程度或不确定性。
    numEntries = len(dataSet)  # 数据集的条目数
    labelCounts = {}  # 创建一个字典，用于统计每个类别的出现次数

    for featVec in dataSet:
        currentLabel = featVec[-1]  # 获取数据集中每条记录的类别（最后一列）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0  # 初始化类别计数
        labelCounts[currentLabel] += 1  # 统计类别的出现次数

    shannonEnt = 0  # 初始化香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算某个类别的概率
        shannonEnt -= prob * log(prob, 2)  # 根据香农熵公式计算并累加
    return shannonEnt  # 返回香农熵


def createDataSet1(path):
    # 从Excel文件中读取数据并生成数据集和标签
    workbook = openpyxl.load_workbook(path)  # 打开Excel文件
    sheet = workbook.active  # 获取活动工作表
    dataSet = []  # 初始化数据集
    labels = [cell.value for cell in sheet[1][1:]]  # 获取特征标签（第一行的值，去掉第一列）

    for row in sheet.iter_rows(min_row=2):
        dataSet.append([cell.value for cell in row[1:]])  # 逐行读取数据并加入数据集

    return dataSet, labels  # 返回数据集和标签


def splitDataSet(dataSet, axis, value):
    # 按照给定特征（axis）和值（value）划分数据集
    retDataSet = []  # 初始化划分后的数据集
    for featVec in dataSet:
        if featVec[axis] == value:  # 如果该特征的值等于指定的值
            reducedFeatVec = featVec[:axis]  # 取该特征前的部分
            reducedFeatVec.extend(featVec[axis+1:])  # 加上该特征后的部分
            retDataSet.append(reducedFeatVec)  # 将处理后的数据加入返回的数据集中
    return retDataSet  # 返回划分后的数据集


def chooseBestFeatureToSplit(dataSet):
    # 选择最优的特征来划分数据集
    numFeatures = len(dataSet[0]) - 1  # 特征的数量（除去最后一列类别标签）
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的原始香农熵
    bestInfoGain = 0  # 初始化最优信息增益
    bestFeature = -1  # 初始化最优特征的索引

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 获取数据集中某特征的所有值
        uniqueVals = set(featList)  # 获取特征值的唯一值集合
        newEntropy = 0  # 初始化新的熵值
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  # 按当前特征的每个唯一值划分数据集
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 累加各子集的熵值
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        if (infoGain > bestInfoGain):  # 如果当前信息增益大于最优信息增益
            bestInfoGain = infoGain  # 更新最优信息增益
            bestFeature = i  # 更新最优特征索引
    return bestFeature  # 返回最优特征的索引


def majorityCnt(classList):
    # 统计分类后的类别，返回出现次数最多的类别
    classCount = {}  # 初始化类别计数字典
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0  # 初始化类别计数
        classCount[vote] += 1  # 统计类别出现次数
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 按类别出现次数排序，降序
    return sortedClassCount[0][0]  # 返回出现次数最多的类别


def createTree(dataSet, labels):
    # 构建决策树
    classList = [example[-1] for example in dataSet]  # 获取数据集中所有的类别标签
    if classList.count(classList[0]) == len(classList):
        # 如果所有类别标签相同，返回该类别
        return classList[0]
    if len(dataSet[0]) == 1:
        # 如果没有特征可分，返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征
    bestFeatLabel = labels[bestFeat]  # 获取最优特征的标签
    myTree = {bestFeatLabel: {}}  # 初始化决策树字典
    del (labels[bestFeat])  # 删除已使用的特征标签
    featValues = [example[bestFeat] for example in dataSet]  # 获取最优特征的所有值
    uniqueVals = set(featValues)  # 获取最优特征的唯一值集合
    for value in uniqueVals:
        subLabels = labels[:]  # 复制标签，防止原标签列表被修改
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
        # 递归创建子树
    return myTree  # 返回构建的决策树


if __name__ == '__main__':
    path = 'data.xlsx'  # Excel文件路径
    dataSet, labels = createDataSet1(path=path)  # 读取数据集和标签
    print(createTree(dataSet, labels))  # 构建并输出决策树
