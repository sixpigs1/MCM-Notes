from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# 生成数据集
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# 定义更深的多层感知机模型


class DeepMLP(nn.Module):
    def __init__(self):
        super(DeepMLP, self).__init__()
        self.layer1 = nn.Linear(2, 5)  # 第1隐藏层：输入2个特征，输出5个神经元
        self.layer2 = nn.Linear(5, 5)  # 第2隐藏层：输入5个神经元，输出5个神经元
        self.layer3 = nn.Linear(5, 5)  # 第3隐藏层：输入5个神经元，输出5个神经元
        self.output = nn.Linear(5, 1)  # 输出层：输入5个神经元，输出1个神经元
        self.relu = nn.ReLU()          # ReLU激活函数
        self.sigmoid = nn.Sigmoid()    # Sigmoid激活函数，用于输出层

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


# 创建模型实例
model = DeepMLP()

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 样例输出
model.eval()
sample_input = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
sample_output = model(sample_input)
print("样例输入:", sample_input)
print("样例输出:", sample_output)

'''
# 输出部分数据以供查看
print("Sample data points:")
print(X[:10])  # 输出前10个数据点
print("Sample labels:")
print(y[:10])  # 输出前10个标签
'''
