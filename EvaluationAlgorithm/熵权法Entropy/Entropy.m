R = [50 30 20 40;
     70 60 90 50;
     80 45 70 60];
%此样例中有三个对象四个评价指标
weights = EntropyWeight(R);
disp(weights)


function weights = EntropyWeight(R)
% 熵权法求指标权重
% R为输入矩阵, 行表示不同的对象，列表示不同的指标
% 返回权重向量 weights，表示每个指标的重要性权重

[rows, cols] = size(R);  % 获取输入矩阵的大小，rows为对象个数，cols为指标个数
k = 1 / log(rows);       % 常数k，用于计算熵值时的归一化系数

f = zeros(rows, cols);   % 初始化fij，fij表示第i个对象在第j个指标上的归一化值
sumBycols = sum(R, 1);   % 计算每一列（即每个指标）的总和，得到1*cols的行向量

% 计算fij，f(i,j)表示第i个对象在第j个指标上的值占该指标总和的比例
for i = 1:rows
    for j = 1:cols
        f(i,j) = R(i,j) / sumBycols(1,j);
    end
end

lnfij = zeros(rows, cols);  % 初始化lnfij，lnfij表示fij的对数值
% 计算lnfij，如果fij为0，则ln(fij)定义为0，避免对数无穷大的情况
for i = 1:rows
    for j = 1:cols
        if f(i,j) == 0
            lnfij(i,j) = 0;  % 当fij为0时，lnfij也设置为0
        else
            lnfij(i,j) = log(f(i,j));  % 计算fij的自然对数
        end
    end
end

Hj = -k * (sum(f .* lnfij, 1));  % 计算每个指标的熵值Hj
weights = (1 - Hj) / (cols - sum(Hj));  % 计算每个指标的权重
end
