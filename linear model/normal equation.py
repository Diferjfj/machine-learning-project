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
##添加偏置
X = np.c_[np.ones(X.shape[0]), X]
##正规方程
theta=np.linalg.pinv(X.T @ X)@ X.T @ y
print(theta)
##计算误差

# 对训练数据进行预测
predictions = X @ theta
mse = np.mean((predictions - y) ** 2)
rmse = np.sqrt(mse)
print(f"均方根误差(RMSE): ${rmse:.2f}")
# 显示前5个预测值与实际值的对比
print("\n前5个预测对比:")
for i in range(min(5, len(y))):
    print(f"实际: ${y.iloc[i]:,.2f}, 预测: ${predictions[i]:,.2f}, 差异: ${abs(y.iloc[i]-predictions[i]):,.2f},百分比：{(y.iloc[i]-predictions[i])/y.iloc[i]:,.2f}")
