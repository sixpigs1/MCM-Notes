import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 生成示例时间序列数据
np.random.seed(42)
n = 120  # 例如120个月的数据
data = np.random.randn(n).cumsum() + 10  # 随机数据生成累计和，并加上一个趋势
dates = pd.date_range(start='2010-01-01', periods=n, freq='M')
time_series = pd.Series(data, index=dates)

# 将数据缩放到0和1之间
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# 准备数据集


def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


time_step = 10
X, y = create_dataset(data_scaled, time_step)

# 将数据重塑为LSTM的输入格式 [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# 创建LSTM模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X, y, epochs=50, batch_size=32, verbose=2)

# 使用模型进行预测
train_predict = model.predict(X)

# 生成未来预测
future_steps = 4  # 预测未来24个月
last_sequence = data_scaled[-time_step:]
future_predictions = []

for _ in range(future_steps):
    next_pred = model.predict(last_sequence.reshape(1, time_step, 1))
    future_predictions.append(next_pred[0, 0])
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = next_pred

# 反缩放预测结果
train_predict = scaler.inverse_transform(train_predict)
future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1))

# 创建未来日期
future_dates = pd.date_range(
    start=dates[-1] + pd.DateOffset(months=1), periods=future_steps, freq='M')

# 可视化预测结果
plt.figure(figsize=(15, 7))
plt.plot(dates, data, label='Historical Data')
plt.plot(dates[time_step:], train_predict,
         label='Training Predictions', color='red')
plt.plot(future_dates, future_predictions,
         label='Future Predictions', color='green')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('LSTM Prediction: Historical and Future')
plt.legend()
plt.show()

# 打印未来预测值
print("\nFuture Predictions:")
for date, prediction in zip(future_dates, future_predictions):
    print(f"{date.strftime('%Y-%m-%d')}: {prediction[0]:.2f}")
