clc;
clear;

N = 10;  % 样本数量

% 训练样本数据
correctData = [0, 0.2, 0.8, 0, 0, 0, 2, 2];
errorData_ReversePharse = [1, 0.8, 0.2, 1, 0, 0, 2, 2];
errorData_CountLoss = [0.2, 0.4, 0.6, 0.2, 0, 0, 1, 1];
errorData_X = [0.5, 0.5, 0.5, 1, 1, 0, 0, 0];
errorData_Lower = [0.2, 0, 1, 0.2, 0, 0, 0, 0];
errorData_Local_X = [0.2, 0.2, 0.8, 0.4, 0.4, 0, 0, 0];
errorData_Z = [0.53, 0.55, 0.45, 1, 0, 1, 0, 0];
errorData_High = [0.8, 1, 0, 0.8, 0, 0, 0, 0];
errorData_CountBefore = [0.4, 0.2, 0.8, 0.4, 0, 0, 2, 2];
errorData_Local_X1 = [0.3, 0.3, 0.7, 0.4, 0.2, 0, 1, 0];

% 合并数据
sampleData = [correctData; errorData_ReversePharse; errorData_CountLoss; errorData_X; errorData_Lower; errorData_Local_X; errorData_Z; errorData_High; errorData_CountBefore; errorData_Local_X1];

% 类别标签
type1 = 1;               % 正确的波形的类别
type2 = -ones(1, N-2);   % 错误的波形的类别
groups = [type1, type2]';  % 合并所有类别标签

j = 1;
% 交叉验证
for i = 2:10
    tempData = sampleData;
    tempData(i, :) = [];   % 轮流从训练数据中去除一个样本作为测试样本
    svmStruct = svmtrain(tempData, groups);  % 训练支持向量机
    species(j) = svmclassify(svmStruct, sampleData(i, :));  % 分类
    j = j + 1;
end

species;  % 输出分类结果
