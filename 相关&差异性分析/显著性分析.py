import scipy.stats as stats

# 假设有两组数据
group1 = [12.9, 10.2, 7.4, 11.5, 10.8]
group2 = [10.2, 9.5, 10.3, 11.1, 12.2]

# 进行独立样本t检验
t_stat, p_value = stats.ttest_ind(group1, group2)

print(f"T统计量: {t_stat}")
print(f"P值: {p_value}")

# 判断显著性
alpha = 0.05
if p_value <= alpha:
    print("结果显著，拒绝零假设H0")
else:
    print("结果不显著，不能拒绝零假设H0")
