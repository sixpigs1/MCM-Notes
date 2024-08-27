clc,clear  % 清空命令窗口并清除工作区中的变量

fid=fopen('AHP.txt','r');  % 打开名为'txt3.txt'的文件，以只读模式读取内容
n1=6;  % 定义准则的数量（假设为6个准则）
n2=3;  % 定义每个准则下的备选方案的数量（假设为3个方案）
a=[];  % 初始化存储准则层判断矩阵的空矩阵

% 读取准则层判断矩阵
for i=1:n1
    tmp=str2num(fgetl(fid));  % 从文件中读取一行字符串并转换为数字数组
    a=[a;tmp];  % 将读取的行添加到准则层判断矩阵中
end

fgetl(fid)

% 读取方案层判断矩阵
for i=1:n1
    str1=char(['b',int2str(i),'=[];']);  % 构造用于初始化方案层判断矩阵的命令，如'b1=[];'
    str2=char(['b',int2str(i),'=[b',int2str(i),';tmp];']);  % 构造用于追加数据的命令，如'b1=[b1;tmp];'
    eval(str1);  % 执行初始化方案层判断矩阵的命令
    for j=1:n2
        tmp=str2num(fgetl(fid));  % 从文件中读取一行字符串并转换为数字数组
        eval(str2);  % 执行将数据追加到方案层判断矩阵中的命令
    end
    fgetl(fid)
end

ri=[0,0,0.58,0.90,1.12,1.24,1.32,1.41,1.45];  % 一致性指标数组，用于一致性检验

% 计算准则层权重向量及一致性比率CR
[x,y]=eig(a);  % 计算准则层判断矩阵'a'的特征向量和特征值
lamda=max(diag(y));  % 获取最大特征值（对应于最大的特征值，即λ_max）
num=find(diag(y)==lamda);  % 找到最大特征值的位置
w0=x(:,num)/sum(x(:,num));  % 计算对应的权重向量，并归一化
cr0=(lamda-n1)/(n1-1)/ri(n1);  % 计算一致性比率CR

% 计算每个方案层权重向量及一致性比率CR
for i=1:n1
    [x,y]=eig(eval(char(['b',int2str(i)])));  % 计算第i个方案层判断矩阵的特征向量和特征值
    lamda=max(diag(y));  % 获取最大特征值（λ_max）
    num=find(diag(y)==lamda);  % 找到最大特征值的位置
    w1(:,i)=x(:,num)/sum(x(:,num));  % 计算对应的权重向量，并归一化
    cr1(i)=(lamda-n2)/(n2-1)/ri(n2);  % 计算一致性比率CR
end

cr1  % 输出每个方案层的一致性比率CR
ts=w1*w0  % 计算最终的综合得分（即各方案的最终权重向量）
cr=cr1*w0  % 计算最终的一致性比率CR
