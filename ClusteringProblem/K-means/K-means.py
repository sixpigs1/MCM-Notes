import time
import matplotlib.pyplot as plt
import numpy as np


def kmeans_xufive(ds, k):
    """k-means聚类算法

    k       - 指定分簇数量
    ds      - ndarray(m, n), m个样本的数据集, 每个样本n个属性值
    """

    m, n = ds.shape  # m：样本数量，n：每个样本的属性值个数
    result = np.empty(m, dtype=np.int64)  # m个样本的聚类结果
    # 从m个数据样本中不重复地随机选择k个样本作为质心
    cores = ds[np.random.choice(np.arange(m), k, replace=False)]

    while True:  # 迭代计算
        d = np.square(np.repeat(ds, k, axis=0).reshape(m, k, n) - cores)
        # ndarray(m, k)，每个样本距离k个质心的距离，共有m行
        distance = np.sqrt(np.sum(d, axis=2))
        index_min = np.argmin(distance, axis=1)  # 每个样本距离最近的质心索引序号

        if (index_min == result).all():  # 如果样本聚类没有改变
            return result, cores  # 则返回聚类结果和质心数据

        result[:] = index_min  # 重新分类
        for i in range(k):  # 遍历质心集
            items = ds[result == i]  # 找出对应当前质心的子样本集
            cores[i] = np.mean(items, axis=0)  # 以子样本集的均值作为当前质心的位置


def create_data_set(*cores):
    """生成k-means聚类测试用数据集"""

    ds = list()
    for x0, y0, z0 in cores:
        x = np.random.normal(x0, 0.1+np.random.random()/3, z0)
        y = np.random.normal(y0, 0.1+np.random.random()/3, z0)
        ds.append(np.stack((x, y), axis=1))

    return np.vstack(ds)


# 聚类数
k = 4

# 随机数据集（10000点）
ds = create_data_set((0, 0, 2500), (0, 2, 2500), (2, 0, 2500), (2, 2, 2500))

# 样例输入：二维数据集
ds = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0],
    [8.0, 2.0],
    [10.0, 2.0],
    [9.0, 3.0],
])

t0 = time.time()
result, cores = kmeans_xufive(ds, k)
t = time.time() - t0

plt.scatter(ds[:, 0], ds[:, 1], s=1, c=result.astype(np.int64))
plt.scatter(cores[:, 0], cores[:, 1], marker='x', c=np.arange(k))
plt.show()

print("聚类结果:", result)
print("质心位置:", cores)

print(u'使用kmeans_xufive算法，1万个样本点，耗时%f0.3秒' % t)
