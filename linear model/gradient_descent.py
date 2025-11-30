import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=None, lambda_=0.0):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.random_state = random_state
        self.lambda_ = lambda_  # L2 正则强度
        self.theta = None
        self.loss_history = []
        self.is_fitted = False
        self.X_mean = None
        self.X_std = None
    def _to_numpy(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy()
        return np.asarray(X)

    def _add_bias(self, X_np):
        return np.c_[np.ones(X_np.shape[0]), X_np]

    def _init_theta(self, n_features_with_bias):
        np.random.seed(self.random_state)
        self.theta = np.random.normal(0, 0.01, n_features_with_bias)

    # 标准化：只处理原始特征，不包括偏置
    def standard_scale(self, X, fit=False):
        X = X.copy()
        if fit or self.X_mean is None or self.X_std is None:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            self.X_std = np.where(self.X_std == 0, 1, self.X_std)
        return (X - self.X_mean) / self.X_std
    # 模型计算
    def forward(self, X_b):
        return X_b @ self.theta
    def compute_loss(self, preds, y):
        mse = np.mean((preds - y) ** 2)
        ridge = self.lambda_ * np.sum(self.theta[1:] ** 2)  # 偏置不正则化
        return mse + ridge
    def backward(self, X_b, y, preds):
        m = X_b.shape[0]
        errors = preds - y
        grad = (2 / m) * (X_b.T @ errors)
        # L2 正则项（偏置项不参与）
        theta_reg = self.theta.copy()
        theta_reg[0] = 0
        grad += 2 * self.lambda_ * theta_reg
        return grad
    # 训练
    def fit(self, X, y, verbose=True):
        X_np = self._to_numpy(X)
        y_np = self._to_numpy(y).reshape(-1)
        # 先标准化
        X_scaled = self.standard_scale(X_np, fit=True)
        # 再加偏置
        X_b = self._add_bias(X_scaled)
        # 初始化参数
        self._init_theta(X_b.shape[1])
        # 梯度下降
        for i in range(self.n_iter):
            preds = self.forward(X_b)
            loss = self.compute_loss(preds, y_np)
            self.loss_history.append(loss)
            grad = self.backward(X_b, y_np, preds)
            self.theta -= self.lr * grad
            if verbose and (i % 100 == 0 or i == self.n_iter - 1):
                print(f"迭代 {i}: 损失 = {loss:.4f}")
        self.is_fitted = True
        if verbose:
            print(f"训练完成！最终损失 = {self.loss_history[-1]:.4f}")

    # 预测
    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        X_np = self._to_numpy(X)
        # 标准化
        X_scaled = self.standard_scale(X_np, fit=False)
        # 加偏置
        X_b = self._add_bias(X_scaled)
        return self.forward(X_b)

    # 可视化
    def plot_loss_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel("迭代次数")
        plt.ylabel("损失 (MSE + Ridge)")
        plt.title("训练损失历史")
        plt.grid(True)
        plt.show()
if __name__ == "__main__":
    data = pd.read_csv("housing.csv")
    #one-hot 处理
    data = pd.get_dummies(data, columns=["ocean_proximity"]).astype(float)
    # 处理缺失值与无穷大
    X = data.drop("median_house_value", axis=1).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    y = data["median_house_value"]

    # 自定义 train_test_split
    def train_test_split(X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        n = len(X)
        n_test = int(n * test_size)
        idx = np.random.permutation(n)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 创建模型
    model = LinearRegressionGD(
        learning_rate=0.1,
        n_iterations=1000,
        random_state=42,
        lambda_=0.01   # L2 正则化强度，可调
    )

    model.fit(X_train, y_train)

    # 预测
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # 评估
    train_rmse = np.sqrt(np.mean((train_pred - y_train.to_numpy()) ** 2))
    test_rmse = np.sqrt(np.mean((test_pred - y_test.to_numpy()) ** 2))

    print("\n模型评估：")
    print(f"训练集 RMSE: {train_rmse:.2f}")
    print(f"测试集 RMSE: {test_rmse:.2f}")
    model.plot_loss_history()
    print("\n前 5 个测试集预测：")
    for actual, pred in list(zip(y_test.to_numpy(), test_pred))[:5]:
        err = abs(actual - pred)
        pct = err / actual * 100
        print(f"实际 ${actual:,.0f} | 预测 ${pred:,.0f} | 误差 ${err:,.0f} | {pct:.2f}%")
