import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def logistic(x):
    return 1 / (1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))

class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        # 输入层到隐藏层的权重初始化
        self.weights.append((2 * np.random.random((layers[0], layers[1])) - 1) * 0.25)
        # 隐藏层到输出层的权重初始化
        self.weights.append((2 * np.random.random((layers[1], layers[2])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        a = x
        for l in range(len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

# Load dataset
data = pd.read_csv(r'C:\Users\mtudou\Desktop\float\data_cleaned.csv')

# 删除时间列（假设时间列是第一列，列名为'NewDateTime'）
if 'NewDateTime' in data.columns:
    data = data.drop(columns=['NewDateTime'])

# 确保所有列都是数值类型
data = data.apply(pd.to_numeric, errors='coerce')

# 检查数据集是否为空
if data.empty:
    raise ValueError("数据集为空，请检查数据预处理步骤。")

# 检查是否存在缺失值
if data.isnull().values.any():
    # 填充缺失值，例如用均值填充
    data = data.fillna(data.mean())

# Assuming the last column is the target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Data preprocessing
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the neural network
nn = NeuralNetwork([X_train.shape[1], 100, 1], 'tanh')
print("开始训练神经网络...")
nn.fit(X_train, y_train, learning_rate=0.01, epochs=10000)
print("神经网络训练完成。")

# Test the neural network
print("\n使用测试集进行评估")
predictions = []
for i in range(X_test.shape[0]):
    prediction = nn.predict(X_test[i])
    predictions.append(prediction)
predictions = np.array(predictions)

# Calculate performance metrics
predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1))
y_test = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test, predictions)
r_squared = r2_score(y_test, predictions)

print(f"\n均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R²): {r_squared:.4f}")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, c='blue', label='预测值')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='理想情况')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('神经网络预测结果')
plt.legend()
plt.show()