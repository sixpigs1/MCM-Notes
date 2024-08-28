%% 初始化工作空间
clc             % 清除命令窗口
clear all       % 清除工作空间中的所有变量
close all       % 关闭所有打开的图形窗口

%% 载入数据
load fisheriris % 加载鸢尾花数据集

%% 二维数组
% 花瓣长度和花瓣宽度散点图(真实标记)
figure; 
speciesNum = grp2idx(species); % 将类别标签转换为数值型（1, 2, 3）
gscatter(meas(:,3),meas(:,4),speciesNum,['r','g','b']) % 绘制散点图，按真实标签着色
xlabel('花瓣长度'); 
ylabel('花瓣宽度');
title('真实标记');
set(gca,'FontSize',12);        % 设置坐标轴字体大小
set(gca,'FontWeight','bold');  % 设置坐标轴字体加粗

% 花瓣长度和花瓣宽度散点图(无标记)
figure;
scatter(meas(:,3),meas(:,4),150,'.'); % 绘制无标签的散点图
xlabel('花瓣长度');
ylabel('花瓣宽度');
title('无标记');
set(gca,'FontSize',12);
set(gca,'FontWeight','bold')

%% 层次聚类
data = [meas(:,3),meas(:,4)]; % 选择用于聚类的特征：花瓣长度和花瓣宽度
datalink = linkage(data,'average','euclidean'); 
% 使用average链接方法和欧氏距离计算样本之间的距离并进行层次聚类

% 绘制树状图
figure;
dendrogram(datalink,10); % 绘制树状图，仅显示10个节点
title('树状图 10节点');
figure;
dendrogram(datalink,0); % 绘制完整的树状图
title('树状图 所有节点');
xtickangle(-45) % 调整x轴标签角度
set(gca,'fontsize',5); % 设置坐标轴字体大小

%% 分割方式1：距离阈值
T1 = cluster(datalink,'cutoff',1.2,'Criterion','distance'); 
% 按距离进行切割，使用距离阈值1.2进行簇划分

% 标号调整
cen = [mean(data(T1==1,:));... % 计算每个簇的中心点
  mean(data(T1==2,:));...
  mean(data(T1==3,:))];
dist = sum(cen.^2,2); % 计算中心点与原点的距离平方和
[dump,sortind] = sort(dist,'ascend'); % 按距离从小到大排序
newT1 = zeros(size(T1)); 
for i=1:3
  newT1(T1==i)=find(sortind==i); % 调整簇标签，使其与原始类别对应
end

%% 分割方式2:群数目
T2 = cluster(datalink,'maxclust',3); % 指定最大簇数为3进行聚类

% 标号调整
cen = [mean(data(T2==1,:));... % 计算每个簇的中心点
  mean(data(T2==2,:));...
  mean(data(T2==3,:))];
dist = sum(cen.^2,2); % 计算中心点与原点的距离平方和
[dump,sortind] = sort(dist,'ascend'); % 按距离从小到大排序
newT2 = zeros(size(T2));
for i = 1:3
  newT2(T2==i)=find(sortind==i); % 调整簇标签，使其与原始类别对应
end

% 花瓣长度和花瓣宽度的散点图(真实标记：实心圆+kmeans分类：圈）
figure;
gscatter(meas(:,3),meas(:,4),speciesNum,['r','g','b']) % 按真实标签绘制散点图（实心圆）
hold on
gscatter(data(:,1),data(:,2),newT2,['r','g','b'],'o',10) % 绘制聚类结果（圆圈）
scatter(cen(:,1),cen(:,2),300,'m*') % 绘制每个簇的中心点（大紫星）
hold off
xlabel('花瓣长度')
ylabel('花瓣宽度')
title('真实标记：实心圆+kmeans分类：圈')
set(gca,'FontSize',12);
set(gca,'FontWeight','bold')
