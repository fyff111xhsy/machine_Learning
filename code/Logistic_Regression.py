"""
  对数几率回归
"""

import numpy as np

# 西瓜数据集 3.0α（密度, 含糖率）
X = np.array([
    [0.697, 0.460],
    [0.774, 0.376],
    [0.634, 0.264],
    [0.608, 0.318],
    [0.556, 0.215],
    [0.403, 0.237],
    [0.481, 0.149],
    [0.437, 0.211],
    [0.666, 0.091],
    [0.243, 0.267],
    [0.245, 0.057],
    [0.343, 0.099],
    [0.639, 0.161],
    [0.657, 0.198],
    [0.360, 0.370],
    [0.593, 0.042],
    [0.719, 0.103]
])

y = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])

# Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 初始化
w = np.zeros(X.shape[1])
b = 0
lr = 0.05
epochs = 20000

# 梯度下降
for _ in range(epochs):
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    dw = np.dot(X.T, (y_hat - y)) / len(y)
    db = np.mean(y_hat - y)
    w -= lr * dw
    b -= lr * db

print("Logistic Regression parameters:")
print("w =", w)
print("b =", b)


"""
  线性判别分析
"""

# 分类别
X1 = X[y == 1]
X0 = X[y == 0]

# 均值
mu1 = np.mean(X1, axis=0)
mu0 = np.mean(X0, axis=0)

# 类内散度
Sw = np.zeros((2,2))
for x in X1:
    Sw += np.outer(x - mu1, x - mu1)
for x in X0:
    Sw += np.outer(x - mu0, x - mu0)

# 投影向量
w = np.linalg.inv(Sw).dot(mu1 - mu0)

print("LDA projection vector w =", w)