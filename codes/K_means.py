"""
  K均值算法
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# 生成模拟数据
def create_data(num_points, num_clusters, cluster_spread):
    data = []
    centers = []
    for _ in range(num_clusters):
        center = (random.uniform(-10, 10), random.uniform(-10, 10))
        centers.append(center)
        for _ in range(num_points // num_clusters):
            point = (np.random.normal(center[0], cluster_spread), 
                     np.random.normal(center[1], cluster_spread))
            data.append(point)
    return np.array(data), np.array(centers)

# K均值算法实现
def k_means(data, k, max_iters=100):
    # 随机初始化质心
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(max_iters):
        # 分配簇
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 更新质心
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # 检查收敛
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# 可视化结果
def plot_clusters(data, centroids, labels):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    data, true_centers = create_data(500, 5, 1)
    centroids, labels = k_means(data, true_centers.shape[0])
    plot_clusters(data, centroids, labels)
