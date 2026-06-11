import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# 1️⃣ 生成二维二分类数据
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# 2️⃣ 定义弱学习器
base_tree = DecisionTreeClassifier(max_depth=5, random_state=42)

# 3️⃣ Bagging
bagging = BaggingClassifier(estimator=base_tree,
                            n_estimators=50,
                            oob_score=True,
                            random_state=42)
bagging.fit(X, y)
print("Bagging OOB Score:", bagging.oob_score_)

# 4️⃣ 随机森林
rf = RandomForestClassifier(n_estimators=50,
                            max_depth=5,
                            oob_score=True,
                            random_state=42)
rf.fit(X, y)
print("Random Forest OOB Score:", rf.oob_score_)

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

# 6️⃣ 可视化
plot_decision_boundary(bagging, X, y, "Bagging Classifier Decision Boundary")
plot_decision_boundary(rf, X, y, "Random Forest Decision Boundary")
