# coding:utf-8
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(filename):
    """
    从指定文件中加载数据集。
    文件格式：每行由两个特征和一个标签组成，特征和标签用逗号分隔。
    """
    dataMat = []
    labelMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split(',')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    """
    随机选择一个不同于 i 的索引 j。
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    """
    将 alpha 值限制在区间 [L, H] 之间。
    """
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    实现SMO算法的简化版，用于训练支持向量机模型。
    输入：数据集 dataMatIn，类别标签 classLabels，常数 C，容错率 toler，最大迭代次数 maxIter。
    输出：目标 b 和优化后的参数 alphas。
    """
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = dataMatrix.shape
    alphas = mat(zeros((m, 1)))
    Iter = 0
    while Iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # 计算预测值和误差
            fXi = float(multiply(alphas, labelMat).T *
                        dataMatrix * dataMatrix[i, :].T) + b
            Ei = fXi - float(labelMat[i])
            # 判断是否需要调整 alpha 值
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T *
                            dataMatrix * dataMatrix[j, :].T) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T \
                    - dataMatrix[i, :] * dataMatrix[i, :].T \
                    - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * \
                    labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei \
                     - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T \
                     - labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[j, :] * dataMatrix[i, :].T
                b2 = b - Ej \
                     - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T \
                     - labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" %
                      (Iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            Iter += 1
        else:
            Iter = 0
        print("iteration number: %d" % Iter)
    return b, alphas


def show(dataArr, labelArr, alphas, b):
    """
    绘制数据点和分类超平面。
    """
    for i in range(len(labelArr)):
        if labelArr[i] == -1:
            plt.plot(dataArr[i][0], dataArr[i][1], 'or')  # 类别 -1 的点用红色圆圈表示
        elif labelArr[i] == 1:
            plt.plot(dataArr[i][0], dataArr[i][1], 'Dg')  # 类别 1 的点用绿色菱形表示
    c = sum(multiply(multiply(alphas.T, mat(labelArr)), mat(dataArr).T), axis=1)
    minY = min(m[1] for m in dataArr)
    maxY = max(m[1] for m in dataArr)
    plt.plot([sum((- b - c[1] * minY) / c[0]),
             sum((- b - c[1] * maxY) / c[0])], [minY, maxY])  # 分隔超平面
    plt.plot([sum((- b + 1 - c[1] * minY) / c[0]),
             sum((- b + 1 - c[1] * maxY) / c[0])], [minY, maxY])  # 支持向量上界
    plt.plot([sum((- b - 1 - c[1] * minY) / c[0]),
             sum((- b - 1 - c[1] * maxY) / c[0])], [minY, maxY])  # 支持向量下界
    plt.show()
