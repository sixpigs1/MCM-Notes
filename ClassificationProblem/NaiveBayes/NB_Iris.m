load fisheriris
% 加载Fisher的鸢尾花数据集。数据集包含4个特征和3个标签。
% 4个特征是萼片长度、萼片宽度、花瓣长度和花瓣宽度。
% 3个标签是三种不同的鸢尾花类别（setosa、versicolor、virginica）。

% 显示数据集中不同物种的数量
tabulate(species);
fprintf('Press enter to continue *^v^* ...');
pause();
% tabulate(species)用于显示每种鸢尾花的样本数量。
% pause()用于暂停代码执行，等待用户按回车键继续。

% 为了简化问题，这里只考虑最后两个特征（花瓣长度和花瓣宽度）
X = meas(:,3:4);  % 提取花瓣长度和花瓣宽度作为特征矩阵X
Y = species;      % 标签向量Y

% 训练一个朴素贝叶斯分类模型
model = fitcnb(X,Y);     
% 使用fitcnb函数训练朴素贝叶斯分类器，model是训练后的模型

% 使用训练好的模型进行预测
predicty = model.predict(X);
% 对训练数据X进行预测，得到每个样本的预测标签

% 计算模型的性能指标
conf = confusionmat(predicty, species)
% 计算混淆矩阵conf，用于评估分类结果的准确性
precision = trace(conf) / size(meas,1)
% 计算分类器的精确度，即预测正确的样本数占总样本数的比例

pause();
% pause()用于暂停代码执行，等待用户按回车键继续。

% 显示模型中的分布参数
setosaIndex = strcmp(model.ClassNames, 'setosa');
% 找到类别'setosa'在模型中的索引
model.DistributionParameters{setosaIndex,1};
% 显示'setosa'类别的分布参数

fprintf('The mean is 1.4620 and the standard deviation is 0.1737.')
% 输出'setosa'类别的均值和标准差

pause();
% pause()用于暂停代码执行，等待用户按回车键继续。

% 可视化概率密度轮廓图
figure
gscatter(X(:,1), X(:,2), Y);
% 绘制散点图，根据不同的类别绘制不同的颜色

h = gca;
cxlim = h.XLim;
cylim = h.YLim;
hold on
% 记录当前坐标轴的x轴和y轴范围，并保持图像不变

% 提取模型中的分布参数
Params = cell2mat(model.DistributionParameters);
Mu = Params(2*(1:3)-1, 1:2); % 提取均值向量
Sigma = zeros(2, 2, 3);
for j = 1:3
  Sigma(:,:,j) = diag(Params(2*j, :)).^2; % 创建对角协方差矩阵
  xlim = Mu(j, 1) + 4*[1 -1]*sqrt(Sigma(1, 1, j));
  ylim = Mu(j, 2) + 4*[1 -1]*sqrt(Sigma(2, 2, j));
  ezcontour(@(x1, x2) mvnpdf([x1, x2], Mu(j, :), Sigma(:,:,j)), [xlim ylim])
  % 绘制多元正态分布的等高线
end

h.XLim = cxlim;
h.YLim = cylim;
title('Naive Bayes Classifier -- Fisher''s Iris Data')
xlabel('Petal Length (cm)')
ylabel('Petal Width (cm)')
hold off
