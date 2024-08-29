% 加载数据
%load("data.mat");

% 确保Y是一个列向量
%if size(Y, 2) > 1
%    Y = Y(:, 1);  % 如果Y有多列，只取第一列
%end
%Y = Y(:);  % 确保Y是一个列向量

% X 样例（假设有10个样本，每个样本有3个特征）
X = [
    2.1, 3.5, 1.4;
    1.5, 2.7, 4.9;
    3.3, 1.2, 2.8;
    4.1, 4.5, 3.6;
    1.9, 2.3, 3.2;
    3.8, 3.1, 2.7;
    2.5, 4.8, 1.9;
    1.7, 2.9, 3.4;
    3.6, 1.8, 4.1;
    4.3, 3.7, 2.5
];

% Y 样例（假设第4列是我们要预测的目标变量）
Y = [
    0;
    0;
    0;
    0;
    1;
    0;
    0;
    1;
    0;
    1
];

% 计算训练集的长度,为总数据的90%
trainlength = floor(length(Y) * 0.9);

% 生成随机排列的索引
index = randperm(length(Y));

% 使用随机索引划分训练集
trainX = X(index(1:trainlength), :);
trainY = Y(index(1:trainlength), :);

% 使用剩余的随机索引划分测试集
testX = X(index(trainlength+1:end), :);
testY = Y(index(trainlength+1:end), :);

% 以下三行被注释掉,可能是之前用于分析决策树的代码
%numBranches = @(x)sum(x.IsBranch);
%mdlDefaultNumSplits = cellfun(numBranches, MdlDefault.Trained);
%view(MdlDefault.Trained{1},'Mode','graph');

% 关闭所有图形窗口
close all;

% 循环训练15个不同复杂度的决策树模型
for i = 1:15
    % 训练决策树模型,最大分裂数从7到21
    mdl{i} = fitctree(X, Y, 'MaxNumSplits', i+6);
    
    % 计算交叉验证误差
    classerror(i) = kfoldLoss(crossval(mdl{i}));
    
    % 计算重采样误差(在训练集上的误差)
    resuberror(i) = resubLoss(mdl{i});
end

% 再次关闭所有图形窗口
close all

% 绘制交叉验证误差和重采样误差的对比图
plot(classerror, 'LineWidth', 2);
hold on;
plot(resuberror, 'LineWidth', 2);
xlabel('节点数');
ylabel('误差');
legend('交叉验证误差', '重采样误差');