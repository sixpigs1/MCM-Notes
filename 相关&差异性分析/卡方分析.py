import numpy as np
from scipy import stats

# 观察到的频数
observed = np.array([60, 40])

# 预期的频数（均匀分布）
expected = np.array([50, 50])

# 进行卡方检验
chi2_statistic, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

print("卡方检验结果：")
print(f"卡方统计量（Chi-Square Statistic）: {chi2_statistic:.2f}")
print(f"P值（P-value）: {p_value:.3f}")
