import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 样本数据
X = np.array([
    [2, 85, 1],  # 学习时间, 出勤率, 家庭作业完成情况(1: 是, 0: 否)
    [3, 90, 1],
    [1, 80, 0],
    [4, 95, 1],
    [2, 70, 0],
    [1, 60, 0],
    [5, 100, 1],
    [3, 80, 0],
    [4, 85, 1],
    [2, 75, 1]
])

y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 1])  # 标签 (1: 通过, 0: 不通过)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 初始化随机森林分类器
clf = RandomForestClassifier(n_estimators=10, random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 输出结果
print("预测结果:", y_pred)
print("实际结果:", y_test.tolist())

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)

# 分类报告
print("分类报告:\n", classification_report(y_test, y_pred))
