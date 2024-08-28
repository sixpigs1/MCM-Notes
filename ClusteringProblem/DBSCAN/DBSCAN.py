import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics

# 数据存储的列表
data = []
# 数据文件的名称
file_name = 'DBSCAN.py\data.txt'
# 数据分隔符
split_char = ','

'''
# 读取文件内容，将每一行的数据转换为浮点数数组并添加到data列表中
with open(file_name) as f:
    for line in f.readlines():
        # 移除每一行首尾的空白字符，并根据指定的分隔符分割
        curline = line.strip().split(split_char)
        # 将分割后的字符串列表转换为浮点数数组
        fltline = np.array(list(map(float, curline)))
        # 将浮点数数组添加到数据列表中
        data.append(fltline)

# 将数据列表转换为NumPy数组，以便进行进一步的处理
data = np.array(data)
'''

data = np.array([
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


def create_data_set(*cores):
    # 生成聚类测试用数据集

    ds = list()
    for x0, y0, z0 in cores:
        x = np.random.normal(x0, 0.1+np.random.random()/0.3, z0)
        y = np.random.normal(y0, 0.1+np.random.random()/0.3, z0)
        ds.append(np.stack((x, y), axis=1))

    return np.vstack(ds)


# 随机数据集（10000点）
data = create_data_set((0, 0, 2500), (0, 20, 2500),
                       (20, 0, 2500), (20, 20, 2500))

# 使用DBSCAN算法对数据进行聚类
# eps参数定义了两个点被认为是邻居的最大距离，min_samples定义了形成核心点所需的最小样本数
c_pred = DBSCAN(eps=2, min_samples=14).fit_predict(data)

# 绘制散点图，按聚类结果着色
plt.scatter(data[:, 0], data[:, 1], c=c_pred)
# 显示散点图
plt.show()
