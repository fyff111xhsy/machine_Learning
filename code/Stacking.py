import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# 1️⃣ 生成二维二分类数据
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# 2️⃣ 定义基学习器（Level-0）
base_learners = [
    ('dt', DecisionTreeClassifier(max_depth=3, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# 3️⃣ 定义次级学习器（Level-1）
meta_learner = LogisticRegression()

# 4️⃣ 构造 Stacking 分类器
stacking = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,  # 交叉验证生成 Level-1 数据
    stack_method='predict_proba'  # 输出概率作为次级学习器输入
)
stacking.fit(X, y)
print("Stacking Classifier Accuracy:", stacking.score(X, y))

# 5️⃣ 可视化分类边界函数
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,200),
                         np.linspace(y_min,y_max,200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.show()

# 6️⃣ 可视化分类边界
plot_decision_boundary(stacking, X, y, "Stacking Classifier Decision Boundary")
