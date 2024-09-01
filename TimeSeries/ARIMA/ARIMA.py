import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# 创建一个示例时间序列数据
np.random.seed(42)
n = 120  # 例如120个月的数据
data = np.random.randn(n).cumsum() + 10  # 随机数据生成累计和，并加上一个趋势
dates = pd.date_range(start='2010-01-01', periods=n, freq='M')
time_series = pd.Series(data, index=dates)

# 可视化时间序列数据
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Time Series Data')
plt.title('Monthly Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# 绘制ACF和PACF图，帮助确定AR和MA的阶数
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(time_series, lags=30, ax=plt.gca())
plt.title('ACF Plot')
plt.subplot(122)
plot_pacf(time_series, lags=30, ax=plt.gca())
plt.title('PACF Plot')
plt.show()

# 选择模型参数（p, d, q）并拟合ARIMA模型
# 这里我们使用(p, d, q) = (1, 1, 1)作为示例
p, d, q = 1, 1, 1
model = ARIMA(time_series, order=(p, d, q))
model_fit = model.fit()

# 打印模型总结
print(model_fit.summary())

# 检查模型残差
residuals = model_fit.resid
plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.plot(residuals)
plt.title('Residuals')
plt.subplot(122)
plot_acf(residuals, lags=30, ax=plt.gca())
plt.title('ACF of Residuals')
plt.show()

# 进行Ljung-Box检验
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(lb_test)

# 进行预测
forecast_steps = 12  # 预测未来12个月
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(
    start=time_series.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
forecast_series = pd.Series(forecast, index=forecast_index)

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Historical Data')
plt.plot(forecast_series, color='red', label='Forecast')
plt.title('ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
