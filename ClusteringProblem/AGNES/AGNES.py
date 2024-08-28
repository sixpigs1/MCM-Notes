from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

# 加载鸢尾花数据集
iris = datasets.load_iris()

# iris数据集属性说明：
# 'data': 用于学习的数据
# 'target': 分类标签
# 'target_names': 标签的含义（例如'0'代表setosa，'1'代表versicolor，'2'代表virginica）
# 'feature_names': 特征的含义（例如花萼长度、宽度等）
# 'DESCR': 数据集的详细描述

# 获取数据集中的数据部分（150个样本，4个特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度）
irisdata = iris.data

# 创建层次聚类模型，使用'ward'链接方法，指定聚类簇的数量为3
clustering = AgglomerativeClustering(linkage='ward', n_clusters=3)

# 拟合聚类模型，并返回每个样本的簇标签
res = clustering.fit(irisdata)

# 打印每个簇中样本的数量
print("各个簇的样本数目：")
print(pd.Series(clustering.labels_).value_counts())

# 打印聚类结果与真实标签的混淆矩阵
print("聚类结果：")
print(confusion_matrix(iris.target, clustering.labels_))

# 可视化聚类结果
plt.figure()

# 绘制第一个簇（簇标签为0）的样本，使用红色点表示
d0 = irisdata[clustering.labels_ == 0]
plt.plot(d0[:, 0], d0[:, 1], 'r.')

# 绘制第二个簇（簇标签为1）的样本，使用绿色圆点表示
d1 = irisdata[clustering.labels_ == 1]
plt.plot(d1[:, 0], d1[:, 1], 'go')

# 绘制第三个簇（簇标签为2）的样本，使用蓝色星号表示
d2 = irisdata[clustering.labels_ == 2]
plt.plot(d2[:, 0], d2[:, 1], 'b*')

# 设置图形的x轴和y轴标签以及标题
plt.xlabel("Sepal.Length")  # 花萼长度
plt.ylabel("Sepal.Width")   # 花萼宽度
plt.title("AGNES Clustering")  # 图形标题

# 显示图形
plt.show()
