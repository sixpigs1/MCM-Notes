function K = kernel(X, Y, type)
% 计算核矩阵
% X 和 Y 为样本矩阵
% type 为核函数类型（'linear' 或 'rbf'）

switch type
    case 'linear'
        K = X' * Y;  % 线性核函数
    case 'rbf'
        delta = 5;   % 高斯核函数的参数
        delta = delta * delta;
        XX = sum(X'.*X', 2); % X 的每行的平方和
        YY = sum(Y'.*Y', 2); % Y 的每行的平方和
        XY = X' * Y;  % X 和 Y 的内积
        K = abs(repmat(XX, [1 size(YY,1)]) + repmat(YY', [size(XX,1) 1]) - 2 * XY);
        K = exp(-K ./ delta);  % 计算高斯核
end
