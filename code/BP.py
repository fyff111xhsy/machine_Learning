import numpy as np

# 西瓜数据集 3.0（属性已数值化）
X = np.array([
    [1,1,1,1,1,0],
    [0,1,0,1,1,0],
    [0,1,1,1,1,0],
    [1,1,0,1,1,0],
    [2,1,1,1,1,0],
    [1,2,1,1,2,1],
    [0,2,1,2,2,1],
    [0,2,1,1,2,0],
    [1,2,0,1,2,1],
    [2,2,0,2,2,0],
    [2,1,1,2,1,1],
    [1,1,1,1,2,1],
    [0,1,0,2,1,0],
    [2,1,0,1,1,0],
    [1,2,1,1,2,1],
    [0,2,1,2,2,0],
    [1,1,1,1,1,0]
])

y = np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(-1,1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# 标准BP算法
def standard_bp(X, y, hidden=8, lr=0.1, epochs=3000):
    n, d = X.shape
    W1 = np.random.randn(d, hidden)
    b1 = np.zeros((1, hidden))
    W2 = np.random.randn(hidden, 1)
    b2 = np.zeros((1, 1))

    for _ in range(epochs):
        for i in range(n):
            xi = X[i:i+1]
            yi = y[i:i+1]

            # forward
            z1 = xi @ W1 + b1
            h = sigmoid(z1)
            z2 = h @ W2 + b2
            o = sigmoid(z2)

            # backward
            delta2 = (o - yi) * sigmoid_derivative(z2)
            delta1 = delta2 @ W2.T * sigmoid_derivative(z1)

            # update
            W2 -= lr * h.T @ delta2
            b2 -= lr * delta2
            W1 -= lr * xi.T @ delta1
            b1 -= lr * delta1

    return W1, b1, W2, b2


# 累积BP算法
def batch_bp(X, y, hidden=8, lr=0.05, epochs=3000):
    n, d = X.shape
    W1 = np.random.randn(d, hidden) * 0.1
    b1 = np.zeros((1, hidden))
    W2 = np.random.randn(hidden, 1) * 0.1
    b2 = np.zeros((1, 1))

    for epoch in range(epochs):

        z1 = X @ W1 + b1
        h = sigmoid(z1)
        z2 = h @ W2 + b2
        o = sigmoid(z2)

        delta2 = (o - y) * o * (1 - o)
        delta1 = delta2 @ W2.T * h * (1 - h)

        W2 -= lr * (h.T @ delta2) / n
        W1 -= lr * (X.T @ delta1) / n
        b2 -= lr * np.mean(delta2, axis=0, keepdims=True)
        b1 -= lr * np.mean(delta1, axis=0, keepdims=True)

    return W1, b1, W2, b2

# 动态BP算法
def dynamic_lr_bp_adjusted(X, y, hidden=8, lr=0.3, epochs=50000, patience=10, min_lr=0.001):
    n, d = X.shape
    # 初始化权值幅度增大
    W1 = np.random.randn(d, hidden) * 1.0
    b1 = np.zeros((1, hidden))
    W2 = np.random.randn(hidden, 1) * 1.0
    b2 = np.zeros((1, 1))

    best_loss = float('inf')
    wait = 0

    for epoch in range(epochs):
        dW1 = np.zeros_like(W1)
        db1 = np.zeros_like(b1)
        dW2 = np.zeros_like(W2)
        db2 = np.zeros_like(b2)
        loss = 0

        for i in range(n):
            xi = X[i:i+1]
            yi = y[i:i+1]

            z1 = xi @ W1 + b1
            h = sigmoid(z1)
            z2 = h @ W2 + b2
            o = sigmoid(z2)

            delta2 = (o - yi) * o * (1 - o)
            delta1 = delta2 @ W2.T * h * (1 - h)

            dW2 += h.T @ delta2
            db2 += delta2
            dW1 += xi.T @ delta1
            db1 += delta1

            loss += np.sum((o - yi) ** 2)

        W2 -= lr * dW2 / n
        b2 -= lr * db2 / n
        W1 -= lr * dW1 / n
        b1 -= lr * db1 / n

        loss /= n

        # 动态调整学习率
        if loss < best_loss:
            best_loss = loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                lr = max(lr / 2, min_lr)
                wait = 0

        if epoch % 1000 == 0:
            print(f"epoch {epoch}: loss={loss:.4f}, lr={lr:.4f}")

    out = sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2)
    return out


if __name__ == "__main__":

    # 训练标准BP模型
    W1_s, b1_s, W2_s, b2_s = standard_bp(X, y)
    print("Standard BP Weights and Biases:")
    print("W1:", W1_s[-1])
    print("b1:", b1_s[-1])
    print("W2:", W2_s[-1])
    print("b2:", b2_s[-1])

    # 训练累积BP模型
    print("\nBatch BP Weights and Biases:")
    W1_b, b1_b, W2_b, b2_b = batch_bp(X, y)
    print("W1:", W1_b[-1])
    print("b1:", b1_b[-1])
    print("W2:", W2_b[-1])
    print("b2:", b2_b[-1])

    # 训练动态BP模型
    print("\nDynamic LR BP Predictions:")
    o_adjusted = dynamic_lr_bp_adjusted(X, y)
    print("预测概率:\n", o_adjusted.T)
    print("真实标签:\n", y.T)
