# MCM-Notes

This repository will contain some methods and codes may be used in MCM.

## Evaluation Algorithms 评价算法

### 1. 层次分析法 Analytic Hierarchy Process (AHP)

- [x] MATLAB
- [ ] Python

### 2. 优劣解距离法 Technique for Order Preference by Similarity to Ideal Solution

- [x] MATLAB, 但是正向化部分存在问题，不建议使用
- [x] Python, 没有问题可以使用

### 3. 熵权法

- [x] MATLAB
- [ ] Python

### 4. 模糊综合评价法 Fuzzy Aggregation Method

- [x] MATLAB, 仅给出了一级的样例
- [ ] Python

### 5. 秩和比评价法 Rank Sum Ratio (RSR)

- [x] MATLAB
- [ ] Python

## Time Series Processing 时间序列处理

### 1. 灰色关联度分析 Grey Relation Analysis (GRA)

用于分析序列（通常是时间序列）中各个子序列与母序列之间的相关性。

- [x] MATLAB
- [ ] Python

### 2. ARIMA模型 ARIMA Model

- [ ] MATLAB
- [x] Python

### 3. LSTM长短期记忆网络 Long Short-Term Memory (LSTM)

- [ ] MATLAB
- [x] Python

## Clustering Problem 聚类问题

### 1. K-means 算法 K-means algorithm

- [ ] MATLAB
- [x] Python

### 2. 密度聚类法 具有噪声的基于密度的聚类方法 Density-Based Spatial Clustering of Applications with Noise (DBSCAN algorithm)

- [ ] MATLAB
- [x] Python

### 3. 层次聚类法 Hierarchical Clustering (HC) (AGNES Algorithm)

- [x] MATLAB, 建议使用MATLAB程序, 从data.txt读取数据, 并且绘图种类丰富
- [x] Python

## Classification Problem 分类问题

### 1. K-邻近算法 K-Nearest Neighbors (KNN)

- [ ] MATLAB
- [x] Python

### 2. 决策树算法 Decision Tree (ID3, C4.5, CART)

- [x] MATLAB, CART
- [x] Python, ID3, C4.5

### 3. 支持向量机 Support Vector Machine (SVM)

- [x] MATLAB
- [x] Python

### 4. 朴素贝叶斯算法 Naive Bayes (NB)

- [x] MATLAB, Fisheriris
- [x] Python, Recommended, 给出了三种实现方法 (BernoulliNB, GaussianNB, MultinomialNB)

### 5. 多层感知机 Multi-Layer Perceptron (MLP)

- [ ] MATLAB
- [x] Python

### 6. BP神经网络 Back-Propagation Neural Network (BPNN)

- [ ] MATLAB
- [x] Python

## 集成学习 Ensemble Learning

### 1. Bagging方法 随机森林 Random Forest (RF)

- [ ] MATLAB
- [x] Python

### 2. Boosting方法 Boosting (AdaBoost, Gradient Boosting)

- [ ] MATLAB
- [x] Python, AdaBoost, Gradient Boosting

### 3. Stacking方法 Stacking (Stacked Generalization)

- [ ] MATLAB
- [x] Python

## 数据降维 Data Reduction

### 1. 主成分分析 Principal Component Analysis (PCA)

- [x] MATLAB
- [x] Python

## 回归分析 Regression Analysis

### 1. 多元线性回归 Multiple Linear Regression (MLR)

- [x] MATLAB
- [ ] Python

### 2. 岭回归 Ridge Regression (RR)

- [x] MATLAB, 有bug, py可以懒得de了
- [x] Python

各种不同的回归模型很多，暂时先写这两个，到时候用到了再看。

## 关于数据读写

从excel读写数据

- [x] MATLAB, 读取
- [x] Python
