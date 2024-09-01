import pandas as pd

# 示例数据
data = [
    ['Group1', 12.9],
    ['Group1', 10.2],
    ['Group1', 7.4],
    ['Group2', 10.2],
    ['Group2', 9.5],
    ['Group2', 10.3],
    ['Group1', 11.5],
    ['Group1', 10.8],
    ['Group2', 11.1],
    ['Group2', 12.2]
]

# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=['Group', 'Value'])

# 将Group列转换为数值型，0代表'Group1'，1代表'Group2'
df['Group'] = df['Group'].map({'Group1': 0, 'Group2': 1})

# 计算皮尔森相关系数
pearson_corr = df['Value'].corr(df['Group'])

# 计算斯皮尔曼相关系数
spearman_corr = df['Value'].corr(df['Group'], method='spearman')

print(f'皮尔森相关系数: {pearson_corr}')
print(f'斯皮尔曼相关系数: {spearman_corr}')

# 判断相关性大小
def judge_correlation(corr):
    if abs(corr) < 0.3:
        return "相关性很小"
    elif abs(corr) < 0.5:
        return "相关性中等"
    else:
        return "相关性较大"

print(judge_correlation(pearson_corr), "（皮尔森）")
print(judge_correlation(spearman_corr), "（斯皮尔曼）")
