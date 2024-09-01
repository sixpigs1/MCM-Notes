% 指定Excel文件的路径
filename = 'output.xlsx';

% 指定数据范围，例如 'B2:D10'
range = 'A1:AJ1555';

% 从Excel文件的指定范围读取数据并保存为矩阵
data = readmatrix(filename, 'Range', range);

% 显示读取的数据
disp(data);
