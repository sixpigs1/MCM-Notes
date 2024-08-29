import numpy as np  # 导入NumPy库，用于数值计算


def zeroMean(dataMat):
    """
    该函数将数据的均值移除，使得数据均值为零。

    参数：
    dataMat -- 输入的数据矩阵，行表示样本，列表示特征

    返回：
    newData -- 去均值后的数据矩阵
    meanVal -- 数据的均值
    """
    meanVal = np.mean(dataMat, axis=0)  # 计算每列的均值（每个特征的均值）
    newData = dataMat - meanVal  # 将均值从数据中减去，得到去均值的数据
    return newData, meanVal  # 返回去均值的数据和均值


def pca(dataMat, percentage=0.8):
    """
    该函数实现PCA算法，通过特征值选择前n个主成分来减少数据的维度。

    参数：
    dataMat -- 输入的数据矩阵，行表示样本，列表示特征
    percentage -- 保留的方差比例，默认为0.99，即保留99%的方差

    返回：
    n_eigVect -- 选出的主成分（特征向量）
    lowDDataMat -- 降维后的数据矩阵
    """
    newData, meanVal = zeroMean(dataMat)  # 去均值
    covMat = np.cov(newData, rowvar=0)  # 计算协方差矩阵，rowvar=0表示按列计算协方差
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 计算协方差矩阵的特征值和特征向量
    eigValIndice = np.argsort(eigVals)  # 获取特征值的索引，并按特征值从小到大排序
    n = percentage2n(eigVals, percentage)  # 计算需要保留的主成分数
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 获取最大的n个特征值的索引
    n_eigVect = eigVects[:, n_eigValIndice]  # 获取对应的n个特征向量
    lowDDataMat = newData * n_eigVect  # 将数据映射到新的特征空间（降维）
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据，即将降维后的数据映射回原空间
    return n_eigVect, lowDDataMat  # 返回主成分和降维后的数据


def percentage2n(eigVals, percentage):
    """
    该函数计算保留一定方差比例所需的主成分数。

    参数：
    eigVals -- 特征值数组
    percentage -- 方差比例，例如0.99

    返回：
    num -- 保留的主成分数
    """
    sortArray = np.sort(eigVals)  # 将特征值从小到大排序
    sortArray = sortArray[-1::-1]  # 逆转数组，即从大到小排序
    arraySum = sum(sortArray)  # 特征值的总和
    tmpSum = 0  # 临时和
    num = 0  # 主成分数
    for i in sortArray:
        tmpSum += i  # 累加特征值
        num += 1  # 计数
        if tmpSum >= arraySum * percentage:  # 判断累计方差是否达到指定比例
            return num  # 返回所需的主成分数


# 从txt文件读取数据
dataMat = np.loadtxt("data.txt", delimiter=",",
                     skiprows=0)  # 从文件读取数据，假设数据以逗号分隔

coeff, lowDDataMat = pca(dataMat)  # 执行PCA算法
print(coeff)  # 打印主成分（特征向量）
print('\n')
print(lowDDataMat)  # 打印降维后的数据
