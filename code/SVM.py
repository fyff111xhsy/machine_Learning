import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 训练并评估 SVM 模型
def train_and_evaluate_svm(X, y, kernel):
    # 数据标准化（SVM 必须）
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    svm = SVC(kernel=kernel, C=1.0, gamma='scale')
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return acc, svm.n_support_


# 数据集1
ir = datasets.load_iris()
X = ir.data
y = ir.target

# 只取前两类
X = X[y < 2]
y = y[y < 2]

acc_linear, sv_linear = train_and_evaluate_svm(X, y, kernel='linear')
acc_rbf, sv_rbf = train_and_evaluate_svm(X, y, kernel='rbf')

print("Iris 数据集结果：")
print(f"线性核 SVM 准确率: {acc_linear:.4f}, 支持向量数: {sv_linear}")
print(f"高斯核 SVM 准确率: {acc_rbf:.4f}, 支持向量数: {sv_rbf}")

# 数据集2
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

acc_linear, sv_linear = train_and_evaluate_svm(X, y, kernel='linear')
acc_rbf, sv_rbf = train_and_evaluate_svm(X, y, kernel='rbf')

print("\nBreast Cancer 数据集结果：")
print(f"线性核 SVM 准确率: {acc_linear:.4f}, 支持向量数: {sv_linear}")
print(f"高斯核 SVM 准确率: {acc_rbf:.4f}, 支持向量数: {sv_rbf}")
