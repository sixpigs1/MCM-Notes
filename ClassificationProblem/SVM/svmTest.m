function result = svmTest(svm, Xt, Yt, kertype)
% 使用训练好的 SVM 进行测试
% svm 为训练得到的支持向量机模型
% Xt 为测试样本
% Yt 为测试标签
% kertype 为核函数类型

temp = (svm.a' .* svm.Ysv) * kernel(svm.Xsv, svm.Xsv, kertype);
total_b = svm.Ysv - temp;
b = mean(total_b);  % 计算偏置项 b
w = (svm.a' .* svm.Ysv) * kernel(svm.Xsv, Xt, kertype);
result.score = w + b;  % 计算预测分数
Y = sign(w + b);       % 预测类别
result.Y = Y;
result.accuracy = size(find(Y == Yt)) / size(Yt);  % 计算准确率
end
