import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# 生成数据
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# DBSCAN 聚类
db = DBSCAN(eps=0.3, min_samples=5)
labels = db.fit_predict(X)

# 可视化
plt.scatter(X[:,0], X[:,1], c=labels, cmap='coolwarm', edgecolor='k')
plt.title("DBSCAN Clustering")
plt.show()
