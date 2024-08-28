%% 初始化工作空间
clc
clear all
close all

%% 从txt文件中读取数据
filename = 'data.txt'; % 定义文件名
data = load(filename); % 从txt文件加载数据

% 假设txt文件中数据的排列顺序与fisheriris数据集一致：
% 前四列为特征，最后一列为类别标签
features = data(:, 1:4);   % 前四列为特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
speciesNum = data(:, 5);   % 最后一列为类别标签：1, 2, 3

%% 二维数组
% 花瓣长度和花瓣宽度散点图(真实标记)
figure;
gscatter(features(:,3),features(:,4),speciesNum,['r','g','b'])
xlabel('花瓣长度');
ylabel('花瓣宽度');
title('真实标记');
set(gca,'FontSize',12);
set(gca,'FontWeight','bold');

% 花瓣长度和花瓣宽度散点图(无标记)
figure;
scatter(features(:,3),features(:,4),150,'.');
xlabel('花瓣长度');
ylabel('花瓣宽度');
title('无标记');
set(gca,'FontSize',12);
set(gca,'FontWeight','bold')

%% 层次聚类
data = [features(:,3), features(:,4)];
datalink = linkage(data,'average','euclidean');

% 绘制树状图
figure;
dendrogram(datalink,10);
title('树状图 10节点');
figure;
dendrogram(datalink,0);
title('树状图 所有节点');
xtickangle(-45)
set(gca,'fontsize',5);

%% 分割方式1：距离阈值
T1 = cluster(datalink,'cutoff',1.2,'Criterion','distance');

% 标号调整
cen = [mean(data(T1==1,:));...
  mean(data(T1==2,:));...
  mean(data(T1==3,:))];
dist = sum(cen.^2,2);
[dump,sortind] = sort(dist,'ascend');
newT1 = zeros(size(T1));
for i=1:3
  newT1(T1==i)=find(sortind==i);
end

%% 分割方式2:群数目
T2 = cluster(datalink,'maxclust',3);

% 标号调整
cen = [mean(data(T2==1,:));...
  mean(data(T2==2,:));...
  mean(data(T2==3,:))];
dist = sum(cen.^2,2);
[dump,sortind] = sort(dist,'ascend');
newT2 = zeros(size(T2));
for i = 1:3
  newT2(T2==i)=find(sortind==i);
end

% 花瓣长度和花瓣宽度的散点图(真实标记：实心圆+kmeans分类：圈）
figure;
gscatter(features(:,3),features(:,4),speciesNum,['r','g','b'])
hold on
gscatter(data(:,1),data(:,2),newT2,['r','g','b'],'o',10)
scatter(cen(:,1),cen(:,2),300,'m*')
hold off
xlabel('花瓣长度')
ylabel('花瓣宽度')
title('真实标记：实心圆+kmeans分类：圈')
set(gca,'FontSize',12);
set(gca,'FontWeight','bold')
