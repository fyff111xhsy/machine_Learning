import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# 1. 构造西瓜 3.0α 示例数据（密度, 含糖率）
# 假设我们只有10个样本
X = np.array([0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243]).reshape(-1,1)
y = np.array([0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267])

# 2. 建立 SVR 模型
# kernel='rbf' 使用高斯核
# C=100, epsilon=0.01 是示例参数，可调节
svr_rbf = SVR(kernel='rbf', C=100, epsilon=0.01, gamma=1)
svr_rbf.fit(X, y)

# 3. 预测并可视化
X_test = np.linspace(0.2, 0.8, 100).reshape(-1,1)
y_pred = svr_rbf.predict(X_test)

plt.scatter(X, y, color='red', label='真实数据')
plt.plot(X_test, y_pred, color='blue', label='SVR预测')
plt.xlabel('密度')
plt.ylabel('含糖率')
plt.title('西瓜 3.0α SVR')
plt.legend()
plt.show()

# 4. 输出支持向量索引
print("支持向量索引:", svr_rbf.support_)
print("支持向量数量:", len(svr_rbf.support_))
