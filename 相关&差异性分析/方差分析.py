import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 单因素方差分析
def one_way_anova():
    print("单因素方差分析：")

    # 数据：假设我们有三个组的数据
    group1 = [23, 20, 22, 21, 24]
    group2 = [30, 29, 31, 32, 28]
    group3 = [25, 27, 26, 24, 28]

    # 进行单因素方差分析
    f_statistic, p_value = stats.f_oneway(group1, group2, group3)

    print(f"F值: {f_statistic:.2f}")
    print(f"P值: {p_value:.3f}")
    print()

# 双因素方差分析
def two_way_anova():
    print("双因素方差分析：")

    # 创建示例数据框
    data = pd.DataFrame({
        'score': [85, 90, 78, 82, 88, 95, 89, 92, 80, 86, 77, 83, 90, 91, 85],
        'method': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],
        'environment': ['X', 'X', 'Y', 'Y', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'X', 'X', 'Y', 'Y', 'X']
    })

    # 进行双因素方差分析
    model = ols('score ~ C(method) * C(environment)', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(anova_table)
    print()

# 执行分析
one_way_anova()
two_way_anova()
