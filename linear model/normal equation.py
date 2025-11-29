import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('housing.csv')

ocean_dummies = pd.get_dummies(data['ocean_proximity'], prefix='ocean')
# 将布尔值转换为整数 (True -> 1, False -> 0)
ocean_dummies = ocean_dummies.astype(int)

# 删除原始的 ocean_proximity 列并添加编码后的列
data = data.drop('ocean_proximity', axis=1)
data = pd.concat([data, ocean_dummies], axis=1)

X = data.drop('median_house_value', axis=1)
y = data['median_house_value']
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())
##分割训练集和测试集
def train_test_split(X,y,test_size,random_state):
    np.random.seed(random_state)
    n=len(X)
    n_test=int(n*test_size)
    indices = np.random.permutation(n)
    test_idx=indices[:n_test]
    train_idx=indices[n_test:]
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    return X_train,X_test,y_train,y_test
##分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
##添加偏置
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
##正规方程
theta=np.linalg.pinv(X_train.T @ X_train)@ X_train.T @ y_train
print(theta)
##计算误差
# 对训练数据进行预测
X_test=np.c_[np.ones(X_test.shape[0]),X_test]
predictions = X_test @ theta
mse = np.mean((predictions - y_test) ** 2)
rmse = np.sqrt(mse)
print(f"均方根误差(RMSE): ${rmse:.2f}")
# 显示前5个预测值与实际值的对比
print("\n前5个预测对比:")
for i in range(min(5, len(y_test))):
    print(f"实际: ${y_test.iloc[i]:,.2f}, 预测: ${predictions[i]:,.2f}, 差异: ${abs(y_test.iloc[i]-predictions[i]):,.2f},百分比：{(y_test.iloc[i]-predictions[i])/y_test.iloc[i]:,.2f}")
