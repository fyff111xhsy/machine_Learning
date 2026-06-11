import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# 1️⃣ 生成二维二分类数据
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# 2️⃣ 定义弱学习器（决策树桩）
stump = DecisionTreeClassifier(max_depth=1, random_state=42)

# 3️⃣ AdaBoost 分类器
adaboost = AdaBoostClassifier(estimator=stump,
                              n_estimators=50,
                              learning_rate=1.0,
                              algorithm='SAMME',
                              random_state=42)
adaboost.fit(X, y)

# 4️⃣ 输出弱分类器权重
print("每个弱分类器的权重:", adaboost.estimator_weights_)
print("adaboost Score:", adaboost.score(X, y))

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
plot_decision_boundary(adaboost, X, y, "AdaBoost Decision Boundary")
