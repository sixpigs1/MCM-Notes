import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 激活函数 - Sigmoid函数


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 激活函数的导数


def sigmoid_derivative(x):
    return x * (1 - x)

# BP神经网络类


class BPNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化BP神经网络的构造函数

        参数:
        - input_size: 输入层的节点数量（即特征的数量）
        - hidden_size: 隐藏层的节点数量
        - output_size: 输出层的节点数量（即类别的数量）
        """
        self.input_size = input_size  # 输入层的节点数量
        self.hidden_size = hidden_size  # 隐藏层的节点数量
        self.output_size = output_size  # 输出层的节点数量

        # 初始化权重，使用均匀分布随机值
        self.weights_input_hidden = np.random.uniform(
            -1, 1, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(
            -1, 1, (self.hidden_size, self.output_size))

    def forward(self, X):
        """
        前向传播函数

        参数:
        - X: 输入数据，形状为 (样本数量, 输入特征数量)

        返回:
        - self.final_output: 网络的最终输出，形状为 (样本数量, 输出类别数量)
        """
        # 计算隐藏层的输入
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        # 计算隐藏层的输出，应用Sigmoid激活函数
        self.hidden_output = sigmoid(self.hidden_input)
        # 计算输出层的输入
        self.final_input = np.dot(
            self.hidden_output, self.weights_hidden_output)
        # 计算输出层的最终输出，应用Sigmoid激活函数
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, learning_rate):
        """
        反向传播函数

        参数:
        - X: 输入数据，形状为 (样本数量, 输入特征数量)
        - y: 实际标签，形状为 (样本数量, 输出类别数量)
        - learning_rate: 学习率，用于调整权重的更新幅度
        """
        # 计算输出层的误差
        output_error = y - self.final_output
        # 计算输出层的梯度
        output_delta = output_error * sigmoid_derivative(self.final_output)

        # 计算隐藏层的误差
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        # 计算隐藏层的梯度
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # 更新隐藏层到输出层的权重
        self.weights_hidden_output += np.dot(
            self.hidden_output.T, output_delta) * learning_rate
        # 更新输入层到隐藏层的权重
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate

    def train(self, X, y, learning_rate=0.1, epochs=10000):
        """
        训练神经网络

        参数:
        - X: 输入数据，形状为 (样本数量, 输入特征数量)
        - y: 实际标签，形状为 (样本数量, 输出类别数量)
        - learning_rate: 学习率，用于调整权重的更新幅度
        - epochs: 训练的轮数，即遍历整个数据集的次数
        """
        for epoch in range(epochs):
            # 前向传播
            self.forward(X)
            # 反向传播
            self.backward(X, y, learning_rate)

    def predict(self, X):
        """
        使用训练好的网络进行预测

        参数:
        - X: 输入数据，形状为 (样本数量, 输入特征数量)

        返回:
        - 预测的类别索引，形状为 (样本数量,)
        """
        # 获取网络的最终输出
        output = self.forward(X)
        # 返回每个样本的预测类别索引
        return np.argmax(output, axis=1)


# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]  # 选择前两个特征（花萼长度和花萼宽度）
y = iris.target

# 对目标值进行独热编码
y_encoded = np.zeros((y.size, y.max() + 1))
y_encoded[np.arange(y.size), y] = 1

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42)

# 初始化BP神经网络
nn = BPNeuralNetwork(input_size=2, hidden_size=5, output_size=3)

# 训练BP神经网络
nn.train(X_train, y_train, learning_rate=0.1, epochs=10000)

# 预测并计算准确率
y_pred = nn.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_labels, y_pred)

# 输出结果
print("Test Accuracy:", accuracy)

# 输出部分数据以供查看
print("Sample data points:")
print(X[:10])  # 输出前10个数据点
print("Sample labels:")
print(y[:10])  # 输出前10个标签
