import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 从Excel文件中读取数据
data = pd.read_excel('data.xlsx')

# 拆分数据集为训练集和测试集
predictors = data.columns[:-1]  # 预测变量（特征）的列名
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    data[predictors],  # 特征数据
    data.Y,            # 目标变量
    test_size=0.2,     # 测试集占比
    random_state=1234  # 随机种子
)

# 标准化特征
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 构造不同的λ值
Lambdas = np.logspace(-5, 2, 200)  # 从10^-5到10^2的200个对数均匀分布的λ值
# 存放每个λ值对应的回归系数
ridge_coefficients = []
mse_train = []
mse_test = []

# 对每个λ值训练岭回归模型，并记录回归系数和MSE
for Lambda in Lambdas:
    ridge = Ridge(alpha=Lambda)  # 创建岭回归模型
    ridge.fit(x_train_scaled, y_train)  # 训练模型
    ridge_coefficients.append(ridge.coef_)  # 记录回归系数

    # 计算训练集和测试集的MSE
    y_train_pred = ridge.predict(x_train_scaled)
    y_test_pred = ridge.predict(x_test_scaled)
    mse_train.append(np.mean((y_train - y_train_pred)**2))
    mse_test.append(np.mean((y_test - y_test_pred)**2))

# 设置全局字体和样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体
plt.rcParams['axes.unicode_minus'] = False  # 支持负号显示
plt.style.use('ggplot')  # 使用ggplot风格

# 创建一个包含两个子图的图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

# 绘制岭迹曲线
for i, coef in enumerate(zip(*ridge_coefficients)):
    ax1.plot(Lambdas, coef, label=f'特征 {predictors[i]}')
ax1.set_xscale('log')  # x轴使用对数尺度
ax1.set_xlabel('Log(Lambda)')  # x轴标签
ax1.set_ylabel('回归系数')  # y轴标签
ax1.set_title('岭回归系数变化曲线')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # 将图例放在图形右侧

# 绘制MSE曲线
ax2.plot(Lambdas, mse_train, label='训练集 MSE', color='blue')
ax2.plot(Lambdas, mse_test, label='测试集 MSE', color='red')
ax2.set_xscale('log')  # x轴使用对数尺度
ax2.set_xlabel('Log(Lambda)')  # x轴标签
ax2.set_ylabel('均方误差 (MSE)')  # y轴标签
ax2.set_title('训练集和测试集的MSE变化曲线')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # 将图例放在图形右侧

# 调整子图之间的间距
plt.subplots_adjust()

# 显示图像
plt.show()
