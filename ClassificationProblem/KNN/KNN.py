import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 从txt文件中读取数据


def load_data_from_txt(file_path, delimiter):
    # 读取文件并将其转化为pandas DataFrame
    data = pd.read_csv(file_path, delimiter=delimiter, header=None)
    return data


# 加载数据
file_path = 'data.txt'  # 假设文件名为data.txt
delimiter = ','         # 假设数据使用逗号分隔
data = load_data_from_txt(file_path, delimiter)

# 分割数据为特征和标签
X = data.iloc[:, :-1].values  # 所有列，除了最后一列是特征
y = data.iloc[:, -1].values   # 最后一列是标签

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)

# 创建KNN模型
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
