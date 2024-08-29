clc,clear
data = load('data.txt');%将原始数据保存在txt文件中
data=zscore(data);     %数据的标准化
r=corrcoef(data);      %计算相关系数矩阵r
%下面利用相关系数矩阵进行主成分分析，vec1的第一列为r的第一特征向量，即主成分的系数
[vec1,lamda,rate]=pcacov(r);                 %lamda为r的特征值，rate为各个主成分的贡献率
f=repmat(sign(sum(vec1)),size(vec1,1),1);    %构造与vec1同维数的元素为±1的矩阵
vec2=vec1.*f;             %修改特征向量的正负号，使得每个特征向量的分量和为正，即为最终的特征向量
num = max(find(lamda>1)); %num为选取的主成分的个数,这里选取特征值大于1的
df=data*vec2(:,1:num);    %计算各个主成分的得分
tf=df*rate(1:num)/100;    %计算综合得分
[stf,ind]=sort(tf,'descend');  %把得分按照从高到低的次序排列
stf=stf'; ind=ind';            %stf为得分从高到低排序，ind为对应的样本编号

disp(df);
disp(tf);
disp(ind);