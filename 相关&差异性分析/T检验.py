from scipy import stats
import numpy as np

# 1. 单样本T检验
print("单样本T检验")

# 数据
sample_data = [510, 515, 520, 505, 510, 505, 500, 515, 520, 505]  # 10个样本数据
known_mean = 500  # 已知的总体平均值

# 计算样本的平均值和标准差
sample_mean = np.mean(sample_data)
sample_std = np.std(sample_data, ddof=1)  # 使用样本标准差
sample_size = len(sample_data)

# 进行单样本T检验
t_statistic, p_value = stats.ttest_1samp(sample_data, known_mean)

print(f"样本平均值: {sample_mean:.2f}")
print(f"样本标准差: {sample_std:.2f}")
print(f"T值: {t_statistic:.2f}")
print(f"P值: {p_value:.3f}")
print()

# 2. 配对样本T检验
print("配对样本T检验")

# 数据
before_treatment = [120, 118, 121, 119, 122, 120, 119, 121, 123, 120]  # 药物服用前的血压
after_treatment = [115, 114, 116, 113, 115, 114, 112, 116, 117, 115]  # 药物服用后的血压

# 进行配对样本T检验
t_statistic, p_value = stats.ttest_rel(before_treatment, after_treatment)

print(f"T值: {t_statistic:.2f}")
print(f"P值: {p_value:.3f}")
print()

# 3. 独立样本T检验
print("独立样本T检验")

# 数据
male_scores = [75, 78, 74, 76, 77, 79, 80, 72, 73, 75, 76, 74, 77, 79, 81]  # 男性的心理测试得分
female_scores = [70, 68, 72, 71, 69, 70, 71, 72, 69, 68, 67, 72, 71, 70, 69]  # 女性的心理测试得分

# 进行独立样本T检验
t_statistic, p_value = stats.ttest_ind(male_scores, female_scores)

print(f"男性组的平均得分: {np.mean(male_scores):.2f}")
print(f"女性组的平均得分: {np.mean(female_scores):.2f}")
print(f"T值: {t_statistic:.2f}")
print(f"P值: {p_value:.3f}")
