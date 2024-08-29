function svm = svmTrain(X, Y, kertype, C)
% 训练支持向量机
% X 为训练样本
% Y 为训练标签
% kertype 为核函数类型
% C 为惩罚参数

options = optimset;        % 创建优化选项
options.LargeScale = 'off';  % 关闭大规模优化
options.Display = 'off';   % 不显示优化过程

n = length(Y);            % 样本数量
H = (Y' * Y) .* kernel(X, X, kertype);  % 构造 Hessian 矩阵

f = -ones(n, 1);  % 目标函数的系数
A = [];
b = [];
Aeq = Y;         % 等式约束
beq = 0;
lb = zeros(n, 1);  % 下界
ub = C * ones(n, 1);  % 上界
a0 = zeros(n, 1);  % 初始解

% 使用 quadprog 进行二次规划
[a, fval, exitflag, output, lambda] = quadprog(H, f, A, b, Aeq, beq, lb, ub, a0, options);

epsilon = 1e-8;                 
sv_label = find(abs(a) > epsilon);  % 寻找支持向量
svm.a = a(sv_label);              % 支持向量的系数
svm.Xsv = X(:, sv_label);         % 支持向量的特征
svm.Ysv = Y(sv_label);            % 支持向量的标签
svm.svnum = length(sv_label);     % 支持向量数量
end
