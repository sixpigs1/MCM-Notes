clc
clear
load accidents % 加载数据（这里假设"accidents"是一个包含相关数据的MAT文件）

% 创建x数据
x = 0: .005: 1; % 从0到1，以0.005为步长生成数据
x = x'; % 将x转置为列向量

% 生成目标数据y
y = sin(x*10*pi) + x; % 目标数据y是x的正弦函数加上x本身

% 不带截距的简单线性回归
b1 = inv(x'*x)*x'*y; % 计算回归系数b1
y1=x*b1; % 使用回归系数计算预测值y1
scatter(x, y) % 绘制原始数据点
hold on
plot(x, y1) % 绘制不带截距的线性回归曲线

% 带截距的线性回归
X=[ones(length(x), 1), x]; % 创建一个包含截距项的设计矩阵X
b2=inv(X'*X)*X'*y; % 计算回归系数b2
y2=X*b2; % 使用回归系数计算预测值y2
plot(x, y2) % 绘制带截距的线性回归曲线

% 加权回归，权重参数sigma=0.08
n=length(x); % 数据点的数量
sigma=0.08; % 权重函数的参数
y3=zeros(n, 1); % 初始化加权回归结果向量y3
for i=1: n
    w=zeros(n, n); % 初始化权重矩阵w
    for j=1: n
        w(j, j)=exp(-(x(j)-x(i)) ^ 2/(2*sigma ^ 2)); % 计算权重
    end
    b3=inv(X'*w*X)*X'*w*y; % 计算加权回归系数b3
    tmp=X*b3; % 计算加权回归预测值
    y3(i)=tmp(i); % 存储预测值
end
plot(x, y3, '*') % 绘制加权回归曲线，权重参数sigma=0.08

% 加权回归，权重参数sigma=0.02
sigma=0.02; % 改变权重函数的参数
y4=zeros(n, 1); % 初始化加权回归结果向量y4
for i=1: n
    w=zeros(n, n); % 初始化权重矩阵w
    for j=1: n
        w(j, j)=exp(-(x(j)-x(i)) ^ 2/(2*sigma ^ 2)); % 计算权重
    end
    b4=inv(X'*w*X)*X'*w*y; % 计算加权回归系数b4
    tmp=X*b4; % 计算加权回归预测值
    y4(i)=tmp(i); % 存储预测值
end
plot(x, y4, '+') % 绘制加权回归曲线，权重参数sigma=0.02

% 设置图形标题和图例
title('several regression relations between', 'fontsize', 20)
legend('data', 'simple linear regression without intercept', ...
    'simple linear regression with an intercept', ...
    'weighted regression with \sigma = 0.08', ...
    'weighted regression with \sigma = 0.02')
