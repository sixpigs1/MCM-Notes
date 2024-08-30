import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

# 生成一个合成数据集，数据点呈圆形分布
X, y = make_circles(factor=0.5, random_state=0, noise=0.05)

# 选择分类方法
print("Select a classification method: 1:BernoulliNB, 2:GaussianNB, or 3:MultinomialNB")
method = input()

# 使用随机树嵌入将数据转换为稀疏矩阵
hasher = RandomTreesEmbedding(n_estimators=10, random_state=0, max_depth=3)
X_transformed = hasher.fit_transform(X)

# 使用TruncatedSVD对稀疏矩阵进行降维，减少到2维
svd = TruncatedSVD(n_components=2)
X_reduced = svd.fit_transform(X_transformed)

# 选择一个朴素贝叶斯分类器
# 你可以分别尝试BernoulliNB, GaussianNB, MultinomialNB

# BernoulliNB:
if (method == "1"):
    nb = BernoulliNB(alpha=0.5, binarize=False,
                     fit_prior=True, class_prior=None)
    nb.fit(X_transformed, y)
# GaussianNB:
# 注意：GaussianNB不支持稀疏矩阵，需要转换为密集矩阵
elif (method == "2"):
    nb = GaussianNB()
    nb.fit(X_transformed.toarray(), y)
# MultinomialNB:
elif (method == "3"):
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_transformed, y)
else:
    print("Invalid input")
    exit()

# 训练一个ExtraTrees分类器作为对比
trees = ExtraTreesClassifier(max_depth=3, n_estimators=10)
trees.fit(X, y)

# 可视化结果
fig = plt.figure(figsize=(9, 8))

# 原始数据的散点图
ax = plt.subplot(221)
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k')
ax.set_title("Original Data")
ax.set_xticks(())
ax.set_yticks(())

# 降维后的数据散点图
ax = plt.subplot(222)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=50, edgecolor='k')
ax.set_title("Truncated SVD reduction(74d->2d)")
ax.set_xticks(())
ax.set_yticks(())

# 创建网格以进行分类结果的可视化
h = 0.01
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 使用训练好的朴素贝叶斯分类器对网格进行预测
transformed_grid = hasher.transform(np.c_[xx.ravel(), yy.ravel()])
if (method == "2"):
    y_grid_pred = nb.predict_proba(transformed_grid.toarray())[:, 1]
else:
    y_grid_pred = nb.predict_proba(transformed_grid)[:, 1]

# 朴素贝叶斯分类器在转换后的数据上的预测结果可视化
ax = plt.subplot(223)
ax.set_title("Naive Bayes on Transformed Data")
ax.pcolormesh(xx, yy, y_grid_pred.reshape(xx.shape))
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k')
ax.set_ylim(-1.4, 1.4)
ax.set_xlim(-1.4, 1.4)
ax.set_xticks(())
ax.set_yticks(())

# 使用训练好的ExtraTrees分类器对网格进行预测
y_grid_pred = trees.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# ExtraTrees分类器的预测结果可视化
ax = plt.subplot(224)
ax.set_title("ExtraTrees prediction")
ax.pcolormesh(xx, yy, y_grid_pred.reshape(xx.shape))
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k')
ax.set_ylim(-1.4, 1.4)
ax.set_xlim(-1.4, 1.4)
ax.set_xticks(())
ax.set_yticks(())

plt.tight_layout()
plt.show()

# 输出部分数据以供查看
print("Sample data points:")
print(X[:10])  # 输出前10个数据点
print("Sample labels:")
print(y[:10])  # 输出前10个标签

# 输出部分数据以供查看
print("Sample data points:")
print(X[:10])  # 输出前10个数据点
print("Sample labels:")
print(y[:10])  # 输出前10个标签
