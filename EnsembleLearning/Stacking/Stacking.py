from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 生成样例数据
X, y = make_classification(n_samples=100, n_features=4,
                           n_informative=2, n_redundant=0, random_state=42)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 定义基学习器
estimators = [
    ('svc', SVC(probability=True, kernel='linear')),
    ('dt', DecisionTreeClassifier(max_depth=3, random_state=42))
]

# 定义元学习器
final_estimator = LogisticRegression()

# 创建并训练 Stacking 模型
model = StackingClassifier(estimators=estimators,
                           final_estimator=final_estimator)
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算并输出准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型在测试集上的准确率: {accuracy:.2f}")

# 输出部分预测结果与真实标签的比较
print("预测结果:", y_pred[:10])
print("真实标签:", y_test[:10])
