% 主函数
clear all;          % 清除所有变量
clc;                % 清空命令窗口

C = 10;             % 设置惩罚参数 C
kertype = 'linear'; % 设置核函数类型为线性

% 生成训练样本
n = 50;             % 每个类别的样本数
randn('state',6);   % 设置随机种子，保证每次生成的随机数相同

% 类别 1 的数据
x1 = randn(2, n);       % 生成 2 行 n 列的随机数据
y1 = ones(1, n);        % 类别标签为 1

% 类别 -1 的数据
x2 = 5 + randn(2, n);   % 生成 2 行 n 列的随机数据，并加上偏移
y2 = -ones(1, n);       % 类别标签为 -1

% 绘制数据点
figure;
plot(x1(1,:), x1(2,:), 'bx', x2(1,:), x2(2,:), 'k.'); 
axis([-3 8 -3 8]);   % 设置坐标轴范围
xlabel('x轴');
ylabel('y轴');
hold on;

% 合并数据
X = [x1, x2];        % 训练样本的特征矩阵，2 行 100 列
Y = [y1, y2];        % 训练样本的标签，1 行 100 列

% 训练支持向量机
svm = svmTrain(X, Y, kertype, C);

% 绘制支持向量
plot(svm.Xsv(1,:), svm.Xsv(2,:), 'ro');

% 测试
[x1, x2] = meshgrid(-2:0.05:7, -2:0.05:7);  % 创建网格数据
[rows, cols] = size(x1);  
nt = rows * cols;                  
Xt = [reshape(x1, 1, nt); reshape(x2, 1, nt)];  % 重塑网格数据为列向量
Yt = ones(1, nt);                       % 初始化测试标签为 1
result = svmTest(svm, Xt, Yt, kertype);  % 测试支持向量机

% 绘制决策边界
Yd = reshape(result.Y, rows, cols);     % 将结果重塑为网格大小
contour(x1, x2, Yd, 'm');               % 绘制等高线
