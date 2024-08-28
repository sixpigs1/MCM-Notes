% 整次秩和比法
clc,clear % 清空命令窗口和工作空间
[data,y] = xlsread('./RSR.xlsx'); % 从Excel文件中读取数据和标签
y = y(2:11,1); % 提取城市名称或对象标签
[score, weight] = shangquan(data); % 使用熵权法计算各指标的权重和各样本的综合得分
data(:,[2,3]) = -data(:,[2,3]); % 数据预处理，将成本型指标（如数据中的第2和第3列）转换为效益型指标（即指标值越大越好）

ra = tiedrank(data); % 编秩，即对每个指标进行排序，得到秩次
[row,col] = size(data); % 获取数据的行列数
RSR = mean(ra, 2) / row; % 计算各评价对象的秩合比，即平均秩次除以总行数
W = repmat(weight, [row,1]); % 将权重向量扩展为与数据行数相同的矩阵
WRSR = sum(ra .* W, 2) / row; % 计算加权秩和比
[sWRSR, ind] = sort(WRSR); % 对加权秩和比排序，并获取排序后的索引
for i = 1:row
    dai = ind;
    ind(i,1) = dai(11-i,1); % 将索引调整为从大到小的顺序
end
p = [1:row] / row; % 计算累计频率
p(end) = 1 - 1 / (4 * row); % 修正最后一个累计频率为1 - 1 / (4n)
probit = norminv(p,0,1) + 5; % 计算标准正态分布的累计分位数并加5进行变换

x = [ones(row,1), probit']; % 构造一元线性回归分析的数据矩阵，第一列为全1
[ab, abint, r, rint, stats] = regress(sWRSR, x); % 对加权秩和比进行回归分析
WRSRfit = ab(1) + ab(2) * probit; % 计算WRSR的估计值
WRSRfit' % 输出估计值
y(ind)' % 输出对应的排序后的标签（城市名称或对象名称）

% 非整次秩和比法
%clc,clear % 清空命令窗口和工作空间
[data,y] = xlsread('./RSR.xlsx'); % 从Excel文件中读取数据和标签
y = y(2:11,1); % 提取城市名称或对象标签
[score, weight] = shangquan(data); % 使用熵权法计算各指标的权重和各样本的综合得分

ra = zeros(10,3); % 初始化秩次矩阵

[row,col] = size(data); % 获取数据的行列数
for i = 1:row
    ra(i,1) = 1 + (row-1) * (data(i,1) - min(data(:,1))) / (max(data(:,1)) - min(data(:,1))); % 计算第1个指标的秩次
end
for i = 1:row
    ra(i,2) = 1 + (row-1) * (max(data(:,2)) - data(i,2)) / (max(data(:,2)) - min(data(:,2))); % 计算第2个指标的秩次
end
for i = 1:row
    ra(i,3) = 1 + (row-1) * (max(data(:,3)) - data(i,3)) / (max(data(:,3)) - min(data(:,3))); % 计算第3个指标的秩次
end
RSR = mean(ra, 2) / row; % 计算各评价对象的秩合比，即平均秩次除以总行数
W = repmat(weight, [row,1]); % 将权重向量扩展为与数据行数相同的矩阵
WRSR = sum(ra .* W, 2) / row; % 计算加权秩和比
[sWRSR, ind] = sort(WRSR); % 对加权秩和比排序，并获取排序后的索引
for i = 1:row
    dai = ind;
    ind(i,1) = dai(11-i,1); % 将索引调整为从大到小的顺序
end
p = [1:row] / row; % 计算累计频率
p(end) = 1 - 1 / (4 * row); % 修正最后一个累计频率为1 - 1 / (4n)
probit = norminv(p,0,1) + 5; % 计算标准正态分布的累计分位数并加5进行变换

x = [ones(row,1), probit']; % 构造一元线性回归分析的数据矩阵，第一列为全1
[ab, abint, r, rint, stats] = regress(sWRSR, x); % 对加权秩和比进行回归分析
WRSRfit = ab(1) + ab(2) * probit; % 计算WRSR的估计值
WRSRfit' % 输出估计值
y(ind)' % 输出对应的排序后的标签（城市名称或对象名称）

% 熵权法 算权重
function [score, weights] = shangquan(x)
% 用熵值法求各指标(列）的权重及各数据行的得分
% x为原始数据矩阵, 一行代表一个样本, 每列对应一个指标

% 数据的归一化处理
X(:, 1) = guiyi(x(:, 1), 1, 0.002, 0.998); % 将第1个指标归一化到[0.002, 0.998]之间
X(:, 2) = guiyi(x(:, 2), 2, 0.002, 0.998); % 将第2个指标归一化到[0.002, 0.998]之间
X(:, 3) = guiyi(x(:, 3), 2, 0.002, 0.998); % 将第3个指标归一化到[0.002, 0.998]之间

% 计算第j个指标下，第i个样本占该指标的比重p(i,j)
for i = 1 : 10
    for j = 1 : 3
        p(i, j) = X(i, j) / sum(X(:, j)); % 计算各样本在每个指标上的占比
    end
end

% 计算第j个指标的熵值e(j)
k = 1 / log(10);
for j = 1 : 3
    e(j) = -k * sum(p(:, j) .* log(p(:, j))); % 计算每个指标的熵值
end

% 计算信息熵冗余度
d = ones(1, 3) - e; 
% 计算各指标的权重w
weights = d ./ sum(d);

% 计算各样本的综合得分
score = 100 * weights * X';
end

function y = guiyi(x, type, y_min, y_max)
% 实现正向或负向指标归一化，返回归一化后的数据矩阵
% x为原始数据矩阵, 每列对应一个指标
% type设定正向指标1, 负向指标2
% y_min, y_max为归一化的区间端点

[n, m] = size(x);
y = zeros(n, m); % 初始化归一化后的矩阵
x_min = min(x);
x_max = max(x);

switch type
    case 1
        for j = 1 : m
            y(:, j) = (y_max - y_min) * (x(:, j) - x_min(j)) / (x_max(j) - x_min(j)) + y_min; % 正向指标归一化
        end
    case 2
        for j = 1 : m
            y(:, j) = (y_max - y_min) * (x_max(j) - x(:,j)) / (x_max(j) - x_min(j)) + y_min; % 负向指标归一化
        end
end
end
