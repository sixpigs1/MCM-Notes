from sklearn.neighbors import KNeighborsClassifier  # 引入KNN分类器
from sklearn.model_selection import train_test_split  # 引入train_test_split函数
from sklearn import datasets  # 从sklearn自带数据库中加载鸢尾花数据
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)

# 创建KNN模型，设置K值为3
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(X_train, y_train)

# 预测测试集
y_pred = knn.predict(X_test)

# 计算准确率
print(y_pred)  # 预测值
print(y_test)  # 真实值
accuracy = accuracy_score(y_test, y_pred)
print(f'模型的准确率: {accuracy:.2f}')
