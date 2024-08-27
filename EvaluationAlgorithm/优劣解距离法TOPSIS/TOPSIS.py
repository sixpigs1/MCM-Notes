import pandas as pd
import numpy as np


def dataDirection_1(datas, offset=0):
    # 极小型指标正向化处理函数
    # datas: 输入的数据列表，代表某一指标的数值
    # offset: 偏移量，默认值为0，用于避免除零的情况

    def normalization(data):
        # 正向化处理，极小型指标的公式为 1 / (data + offset)
        return 1 / (data + offset)

    # 将正向化处理应用到每一个数据点
    return list(map(normalization, datas))


def dataDirection_2(datas, x_min, x_max):
    # 中间型指标正向化处理函数
    # datas: 输入的数据列表，代表某一指标的数值
    # x_min: 指标理想的最小值
    # x_max: 指标理想的最大值

    def normalization(data):
        # 中间型指标正向化处理，分为三个区间
        if data <= x_min or data >= x_max:
            return 0  # 如果数据在理想区间之外，则返回0
        elif data > x_min and data < (x_min + x_max) / 2:
            return 2 * (data - x_min) / (x_max - x_min)  # 左半区间正向化处理
        elif data < x_max and data >= (x_min + x_max) / 2:
            return 2 * (x_max - data) / (x_max - x_min)  # 右半区间正向化处理

    # 将正向化处理应用到每一个数据点
    return list(map(normalization, datas))


def dataDirection_3(datas, x_min, x_max, x_minimum, x_maximum):
    # 区间型指标正向化处理函数
    # datas: 输入的数据列表，代表某一指标的数值
    # x_min: 指标理想区间的最小值
    # x_max: 指标理想区间的最大值
    # x_minimum: 指标的最小允许值
    # x_maximum: 指标的最大允许值

    def normalization(data):
        # 区间型指标正向化处理，分为四个区间
        if data >= x_min and data <= x_max:
            return 1  # 如果数据在理想区间内，返回1
        elif data <= x_minimum or data >= x_maximum:
            return 0  # 如果数据在允许范围之外，返回0
        elif data > x_max and data < x_maximum:
            return 1 - (data - x_max) / (x_maximum - x_max)  # 右边区间处理
        elif data < x_min and data > x_minimum:
            return 1 - (x_min - data) / (x_min - x_minimum)  # 左边区间处理

    # 将正向化处理应用到每一个数据点
    return list(map(normalization, datas))


def topsis(data, weight=None):
    # TOPSIS方法实现函数
    # data: 输入的决策矩阵，每一行代表一个方案，每一列代表一个指标
    # weight: 指标的权重，如果为None则使用熵权法计算权重

    # 归一化处理，将每个指标的数值归一化到0到1之间
    data = data / np.sqrt((data ** 2).sum())

    # 计算正理想解（最大值）和负理想解（最小值）
    Z = pd.DataFrame([data.min(), data.max()], index=['负理想解', '正理想解'])

    # 计算每个方案到正理想解和负理想解的距离
    weight = entropyWeight(data) if weight is None else np.array(weight)
    Result = data.copy()
    Result['正理想解'] = np.sqrt(
        ((data - Z.loc['正理想解']) ** 2 * weight).sum(axis=1))
    Result['负理想解'] = np.sqrt(
        ((data - Z.loc['负理想解']) ** 2 * weight).sum(axis=1))

    # 计算每个方案的综合得分指数
    Result['综合得分指数'] = Result['负理想解'] / (Result['负理想解'] + Result['正理想解'])
    Result['排序'] = Result.rank(ascending=False)['综合得分指数']

    # 返回结果，包括每个方案的综合得分指数和排序
    return Result, Z, weight


data = pd.DataFrame({
    '价格': [2500, 2000, 3000],
    '交货时间': [7, 8, 6],
    '产品质量': [5, 7, 4]
})

data['价格'] = dataDirection_1(data['价格'])
data['交货时间'] = dataDirection_1(data['交货时间'])

result, ideal_solution, weight = topsis(data, weight=[0.4, 0.3, 0.3])
print(result)
