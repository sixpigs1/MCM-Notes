from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 生成样例数据
X, y = make_classification(n_samples=100, n_features=4,
                           n_informative=2, n_redundant=0, random_state=42)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 创建一个简单的决策树弱分类器
base_classifier = DecisionTreeClassifier(max_depth=1)

# 使用 AdaBoost 进行训练
model = AdaBoostClassifier(
    estimator=base_classifier, n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算并输出准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型在测试集上的准确率: {accuracy:.2f}")

# 输出部分预测结果与真实标签的比较
print("预测结果:", y_pred[:10])
print("真实标签:", y_test[:10])

# 输出部分数据以供查看
print("Sample data points:")
print(X[:10])  # 输出前10个数据点
print("Sample labels:")
print(y[:10])  # 输出前10个标签
