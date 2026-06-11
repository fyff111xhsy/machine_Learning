"""
  KNN算法
"""

import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """
        训练KNN模型
        :param X: 训练数据特征，形状为 (n_samples, n_features)
        :param y: 训练数据标签，形状为 (n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        使用KNN模型进行预测
        :param X: 测试数据特征，形状为 (m_samples, n_features)
        :return: 预测标签，形状为 (m_samples,)
        """
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """
        对单个样本进行预测
        :param x: 单个测试样本，形状为 (n_features,)
        :return: 预测标签
        """
        # 计算距离
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # 获取最近的k个邻居的索引
        k_indices = np.argsort(distances)[:self.k]
        # 获取最近的k个邻居的标签
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 返回出现次数最多的标签
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

