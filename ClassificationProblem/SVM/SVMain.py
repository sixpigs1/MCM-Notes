import SVM
from numpy import *

# 从文件 'testSet.txt' 加载数据集
dataArr, labelArr = SVM.loadDataSet('test.txt')

# 使用SMO算法进行训练，参数分别是数据集、类别标签、常数C、容错率和最大迭代次数
b, alphas = SVM.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)  # 参数修改位置在此处

# 绘制分类结果
SVM.show(dataArr, labelArr, alphas, b)
