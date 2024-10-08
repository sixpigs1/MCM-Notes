x = [1 2; 3 4; 5 6; 7 8; 9 10; 11 12; 13 14; 15 16; 17 18; 19 20; 
     21 22; 23 24; 25 26; 27 28; 29 30; 31 32; 33 34; 35 36; 37 38; 39 40;
     41 42; 43 44; 45 46; 47 48; 49 50; 51 52; 53 54; 55 56; 57 58; 59 60;
     61 62; 63 64; 65 66; 67 68; 69 70; 71 72; 73 74; 75 76; 77 78; 79 80];
y = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 
     11; 12; 13; 14; 15; 16; 17; 18; 19; 20;
     21; 22; 23; 24; 25; 26; 27; 28; 29; 30;
     31; 32; 33; 34; 35; 36; 37; 38; 39; 40];
k = [0.1, 1, 10];


b = ridge(y, x, k); 
% 计算岭回归模型。y 是目标变量，x 是特征矩阵，k 是一组岭参数
% b 是每个岭参数下的回归系数矩阵

knum = size(b, 2);
% 计算岭参数的数量

sse = zeros(knum, 1);
% 初始化 SSE（误差平方和）数组

y_gu = zeros(18, knum);
% 初始化预测结果矩阵，假设有 18 个样本，每个样本有 knum 个岭参数下的预测值

for j = 1:knum
    t = 0;
    % 初始化误差平方和变量

    % 此处对具体矩阵进行处理
    % 计算第 j 个岭参数下的误差平方和
    % 这里的处理代码应该包括用第 j 个岭参数的回归系数计算预测值，并与真实值比较，更新 SSE

    sse(j) = t;
    % 将计算出的误差平方和存储到数组中
end

plot(k, sse);
% 绘制岭参数 k 与误差平方和 SSE 的关系图
xlabel('k');
% x 轴标签
ylabel('SSE');
% y 轴标签

b1 = ridge(y, x, 2, 0);
% 使用岭参数 2 计算回归模型，最后一个参数 0 表示不输出标准化系数

% 用岭回归模型求出的函数估计值（向量）
y_gu = [ones(18, 1), x] * b1;
% 对输入特征 x 进行预测，假设 x 有 18 个样本
