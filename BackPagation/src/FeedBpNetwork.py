'''
Created on 2025年5月23日

@author: zhaoyangchun
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm  # 添加tqdm库用于进度条显示
from scipy import stats
import math
import os

# 获取当前文件所在目录的绝对路径（即 src 目录）
current_dir = os.path.dirname(__file__)

# 获取项目根目录（即 src 的上一级目录）
project_root = os.path.dirname(current_dir)

# 构建数据文件的路径
feed_data_file_path = os.path.join(project_root, 'data', 'feed.csv')

# 设置中文显示和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC','Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================
# 1. 激活函数及其导数定义
# ==============================
def leaky_relu(x, alpha=0.01):

    return np.maximum(alpha * x, x)


def leaky_relu_deriv(x, alpha=0.01):

    return np.where(x > 0, 1, alpha)


# ==============================
# 2. BP神经网络类定义
# ==============================
class BPNetwork:
    def __init__(self, layers):
        """初始化网络结构和权重"""
        self.layers = layers
        self.weights = []


        for i in range(len(layers) - 1):
            fan_in, fan_out = layers[i], layers[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(np.random.randn(fan_in, fan_out) * scale)

    def forward(self, x):
        """前向传播"""
        a = x
        for w in self.weights[:-1]:
            a = leaky_relu(np.dot(a, w))
        a = np.dot(a, self.weights[-1])
        return a

    def backward(self, x_batch, y_batch, learning_rate):
        """反向传播更新权重"""
        activations = [x_batch]
        z_values = []


        for w in self.weights[:-1]:
            z = np.dot(activations[-1], w)
            z_values.append(z)
            a = leaky_relu(z)
            activations.append(a)

        z = np.dot(activations[-1], self.weights[-1])
        z_values.append(z)
        activations.append(z)


        deltas = [activations[-1] - y_batch]


        for i in range(len(self.weights) - 2, -1, -1):
            delta = deltas[-1].dot(self.weights[i + 1].T) * leaky_relu_deriv(z_values[i])
            deltas.append(delta)
        deltas.reverse()


        for i in range(len(self.weights)):
            grad = activations[i].T.dot(deltas[i]) / x_batch.shape[0]
            self.weights[i] -= learning_rate * grad

    def fit(self, X, y, initial_lr=0.001, decay=0.0001, epochs=1000, batch_size=32, verbose=True):

        X = np.atleast_2d(X)
        y = np.array(y).reshape(-1, 1)
        n_samples = X.shape[0]


        val_size = int(n_samples * 0.2)
        X_val, y_val = X[:val_size], y[:val_size]
        X_train, y_train = X[val_size:], y[val_size:]


        train_losses = []
        val_losses = []
        learning_rates = []
        best_val_loss = float('inf')
        best_weights = None

        print(f"开始训练...\n样本总数: {n_samples}\n训练集样本: {len(X_train)}\n验证集样本: {len(X_val)}")
        print(f"参数配置：初始学习率={initial_lr}, 衰减率={decay}, 批次大小={batch_size}, 轮次={epochs}")

        start_time = time.time()


        with tqdm(range(epochs), desc="训练进度", unit="epoch") as pbar:
            for epoch in pbar:
                lr = initial_lr / (1 + decay * epoch)
                idx = np.random.permutation(len(X_train))
                X_shuffled = X_train[idx]
                y_shuffled = y_train[idx]

                epoch_loss = 0.0

                for i in range(0, len(X_train), batch_size):
                    X_batch = X_shuffled[i:i + batch_size]
                    y_batch = y_shuffled[i:i + batch_size]
                    self.backward(X_batch, y_batch, lr)

                    batch_pred = self.forward(X_batch)
                    epoch_loss += np.mean((y_batch - batch_pred) ** 2) * len(X_batch)

                epoch_loss /= len(X_train)  #
                val_pred = self.forward(X_val)
                val_loss = np.mean((y_val - val_pred) ** 2)  #


                train_losses.append(epoch_loss)
                val_losses.append(val_loss)
                learning_rates.append(lr)


                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = [w.copy() for w in self.weights]


                pbar.set_postfix({
                    "LR": f"{lr:.6f}",
                    "Train Loss": f"{epoch_loss:.4f}",
                    "Val Loss": f"{val_loss:.4f}",
                    "Best Val": f"{best_val_loss:.4f}"
                })


                if (epoch + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    eta = (epochs - epoch - 1) * (elapsed / (epoch + 1)) if (epoch + 1) != 0 else 0
                    print(
                        f"\nEpoch {epoch + 1:4d}/{epochs} | LR: {lr:.6f} | 训练损失: {epoch_loss:.4f} | 验证损失: {val_loss:.4f} | 最佳验证: {best_val_loss:.4f} | 耗时: {elapsed:.2f}s | ETA: {eta:.2f}s")


        if best_weights:
            self.weights = best_weights
        total_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {total_time:.2f}秒\n最佳验证损失: {best_val_loss:.4f}")

        #
        self.history = {
            'epochs': np.arange(1, epochs + 1),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'learning_rate': learning_rates,
            'best_epoch': np.argmin(val_losses) + 1  # 最佳轮次（从1开始计数）
        }
        return self

    def predict(self, x):
        """预测函数"""
        return self.forward(np.atleast_2d(x)).flatten()


# ==============================
# 3. 数据处理与训练流程
# ==============================
if __name__ == "__main__":
    # 构建 model_output 文件夹的路径（位于项目根目录下）
    output_dir =os.path.join(project_root, 'model_output')
    # 创建 model_output 文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------
    # 3.1 读取与清洗数据
    # ------------------------------
    try:
        data = pd.read_csv(feed_data_file_path)  # 使用 pandas 读取 feed.csv 文件
    except FileNotFoundError:
        print("错误：未找到数据文件feed.csv，请检查路径！")
        exit(1)



    print("\n数据基本信息：")
    data.info()
    print("\n数据前5行：")
    print(data.head())
    print("\n数据列名：")
    print(data.columns.tolist())


    columns_to_drop = []


    if '' in data.columns:
        columns_to_drop.append('')
    elif 'index' in data.columns.str.lower():
        index_col = [col for col in data.columns if col.lower() == 'index'][0]
        columns_to_drop.append(index_col)
    elif '序号' in data.columns:
        columns_to_drop.append('序号')


    if 'date' in data.columns.str.lower():
        date_col = [col for col in data.columns if col.lower() == 'date'][0]
        columns_to_drop.append(date_col)
    elif '日期' in data.columns:
        columns_to_drop.append('日期')


    if columns_to_drop:
        print(f"\n将删除以下列：{columns_to_drop}")
        data = data.drop(columns=columns_to_drop)
    else:
        print("\n未检测到需要删除的冗余列")


    print("\n正在处理数据类型和缺失值...")
    data = data.apply(pd.to_numeric, errors='coerce')  #
    missing_values = data.isna().sum().sum()
    if missing_values > 0:
        print(f"检测到 {missing_values} 个缺失值，使用均值填充")
        data = data.fillna(data.mean())  #
    else:
        print("数据中没有缺失值")

    X = data.iloc[:, :-1].values  #
    y = data.iloc[:, -1].values  #

    if X.size == 0 or y.size == 0:
        print("错误：特征或目标变量为空，请检查数据结构")
        exit(1)

    # ------------------------------
    # ------------------------------
    print("\n正在进行数据标准化...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X)  #
    y = scaler_y.fit_transform(y.reshape(-1, 1))
    y = y.flatten()

    # ------------------------------
    # 3.3 划分训练集与测试集
    # ------------------------------
    print("\n正在划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # ------------------------------
    # 3.4 定义与训练BP网络
    # ------------------------------
    input_dim = X_train.shape[1]
    print(f"\n网络配置：输入特征数={input_dim}")

    nn = BPNetwork(
        layers=[input_dim, 128, 64, 32, 1]  #
    )

    nn.fit(
        X_train, y_train,
        initial_lr=0.003,  # 初始学习率
        decay=0.0001,  # 学习率衰减率
        epochs=10,  # 训练轮次
        batch_size=64,  # 批次大小
        verbose=True
    )

    # ------------------------------
    # 3.5 模型评估
    # ------------------------------
    y_pred = nn.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # 计算评估指标
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    evs = explained_variance_score(y_test_original, y_pred)
    rmspe = np.sqrt(np.mean(np.square(((y_test_original - y_pred) / y_test_original)), axis=0)) * 100
    mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100

    print(f"\n----------------------- 评估结果 -----------------------")
    print(f"测试集样本数: {len(X_test)}")
    print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    print(f"R²: {r2:.4f} | 解释方差: {evs:.4f}")
    print(f"RMSPE: {rmspe:.2f}% | MAPE: {mape:.2f}%")

    # ------------------------------
    # 3.6 可视化分析
    # ------------------------------
    from matplotlib.backends.backend_pdf import PdfPages

    pdf = PdfPages(f'{output_dir}/bp_network_evaluation.pdf')

    # ------------------------------
    # 3.6.1 训练过程可视化
    # ------------------------------
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    plt.plot(nn.history['epochs'], nn.history['train_loss'], label='训练损失', c='blue', alpha=0.7)
    plt.plot(nn.history['epochs'], nn.history['val_loss'], label='验证损失', c='red', alpha=0.7)
    plt.axvline(nn.history['best_epoch'], color='black', linestyle='--', alpha=0.5)
    plt.xlabel('训练轮次')
    plt.ylabel('损失值 (MSE)')
    plt.title('训练与验证损失变化曲线')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(nn.history['epochs'], nn.history['learning_rate'], c='green', alpha=0.6)
    plt.xlabel('训练轮次')
    plt.ylabel('学习率')
    plt.title('学习率衰减过程')
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(nn.history['epochs'], np.log10(nn.history['train_loss']), label='训练损失 (log10)', c='blue', alpha=0.7)
    plt.plot(nn.history['epochs'], np.log10(nn.history['val_loss']), label='验证损失 (log10)', c='red', alpha=0.7)
    plt.axvline(nn.history['best_epoch'], color='black', linestyle='--', alpha=0.5)
    plt.xlabel('训练轮次')
    plt.ylabel('损失值 (log10 MSE)')
    plt.title('对数尺度损失变化曲线')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.subplot(2, 2, 4)
    loss_ratio = np.array(nn.history['val_loss']) / np.array(nn.history['train_loss'])
    plt.plot(nn.history['epochs'], loss_ratio, c='purple', alpha=0.7)
    plt.axvline(nn.history['best_epoch'], color='black', linestyle='--', alpha=0.5)
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('训练轮次')
    plt.ylabel('验证损失 / 训练损失')
    plt.title('损失比值变化曲线（检测过拟合）')
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # ------------------------------
    # 3.6.2 预测性能可视化
    # ------------------------------
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    plt.scatter(y_test_original, y_pred, c='purple', alpha=0.8, edgecolors='white')
    plt.plot([y_test_original.min(), y_test_original.max()],
             [y_test_original.min(), y_test_original.max()],
             'k--', lw=2, alpha=0.7)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'预测值与实际值对比 (R²={r2:.4f})')
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.subplot(2, 2, 2)
    residuals = y_test_original - y_pred
    sns.histplot(residuals, kde=True, color='orange', alpha=0.6)
    plt.axvline(0, color='red', linestyle='--', alpha=0.8)
    plt.xlabel('残差 (实际值-预测值)')
    plt.ylabel('频率')
    plt.title('残差分布直方图')
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.subplot(2, 2, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('残差Q-Q图（检验正态性）')
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.subplot(2, 2, 4)
    sorted_idx = np.argsort(y_test_original)
    plt.scatter(range(len(y_test_original)), y_test_original[sorted_idx] - y_pred[sorted_idx],
                c='blue', alpha=0.7, edgecolors='white')
    plt.axhline(0, color='red', linestyle='--', alpha=0.8)
    plt.xlabel('样本索引（按实际值排序）')
    plt.ylabel('预测误差')
    plt.title('预测误差散点图')
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # ------------------------------
    # 3.6.3 预测性能分析
    # ------------------------------
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    plt.boxplot(residuals, vert=False)
    plt.axvline(0, color='red', linestyle='--', alpha=0.8)
    plt.xlabel('预测误差')
    plt.title('预测误差箱线图')
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.subplot(2, 2, 2)
    n_bins = 5
    y_bins = pd.qcut(y_test_original, n_bins, labels=False)
    bin_errors = []
    bin_labels = []

    for i in range(n_bins):
        mask = (y_bins == i)
        if np.sum(mask) > 0:
            bin_error = np.mean(np.abs(y_test_original[mask] - y_pred[mask]))
            bin_errors.append(bin_error)
            bin_min = np.min(y_test_original[mask])
            bin_max = np.max(y_test_original[mask])
            bin_labels.append(f"{bin_min:.2f}-{bin_max:.2f}")

    plt.bar(bin_labels, bin_errors, color='skyblue', alpha=0.7)
    plt.xlabel('目标值区间')
    plt.ylabel('平均绝对误差')
    plt.title('不同目标值区间的预测精度')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.subplot(2, 2, 3)
    sns.kdeplot(y_test_original, label='实际值分布', color='blue', alpha=0.7)
    sns.kdeplot(y_pred, label='预测值分布', color='red', alpha=0.7)
    plt.xlabel('值')
    plt.ylabel('密度')
    plt.title('预测值与实际值分布对比')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.subplot(2, 2, 4)
    heatmap, xedges, yedges = np.histogram2d(y_test_original, y_pred, bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', alpha=0.8)
    plt.colorbar(label='样本数量')
    plt.plot([y_test_original.min(), y_test_original.max()],
             [y_test_original.min(), y_test_original.max()],
             'r--', lw=2, alpha=0.7)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('预测误差热力图')

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # ------------------------------
    # 3.6.4 特征分析
    # ------------------------------
    if X_train.shape[1] <= 20:
        plt.figure(figsize=(15, 10))

        plt.subplot(1, 2, 1)
        feature_names = data.columns[:-1].tolist()  # 特征名称
        corr_matrix = pd.concat([pd.DataFrame(X_train, columns=feature_names),
                                 pd.DataFrame(y_train, columns=['目标'])], axis=1).corr()

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
        plt.title('特征与目标变量相关性热力图')

        plt.subplot(1, 2, 2)
        feature_importance = np.abs(nn.weights[0]).sum(axis=1)
        feature_importance = feature_importance / np.sum(feature_importance)  # 归一化

        sorted_idx = np.argsort(feature_importance)[::-1]
        sorted_importance = feature_importance[sorted_idx]
        sorted_features = [feature_names[i] for i in sorted_idx]

        plt.barh(range(len(sorted_importance)), sorted_importance, color='skyblue', alpha=0.7)
        plt.yticks(range(len(sorted_importance)), sorted_features)
        plt.xlabel('相对重要性')
        plt.title('基于第一层权重的特征重要性')
        plt.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # ------------------------------
    # 3.6.5 保存评估报告
    # ------------------------------
    pdf.close()
    print(f"\n已生成完整评估报告：{output_dir}/bp_network_evaluation.pdf")


    evaluation_results = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'Explained Variance', 'RMSPE (%)', 'MAPE (%)'],
        'Value': [mse, rmse, mae, r2, evs, rmspe, mape]
    })
    evaluation_results.to_csv(f'{output_dir}/model_evaluation_metrics.csv', index=False)
    print(f"已保存评估指标到：{output_dir}/model_evaluation_metrics.csv")

    results_df = pd.DataFrame({
        'Actual': y_test_original,
        'Predicted': y_pred,
        'Residual': y_test_original - y_pred,
        'Absolute Error': np.abs(y_test_original - y_pred),
        'Percentage Error': np.abs((y_test_original - y_pred) / y_test_original) * 100
    })
    results_df.to_csv(f'{output_dir}/prediction_results.csv', index=False)
    print(f"已保存预测结果到：{output_dir}/prediction_results.csv")

    # ------------------------------
    # 3.6.6 预测结果与实际值对比图
    # ------------------------------
    print("\n正在生成预测结果与实际值对比图...")

    # 创建一个输出目录
    os.makedirs(f'{output_dir}/prediction_plots', exist_ok=True)


    max_points = 10000  #
    if len(y_test_original) > max_points:
        # 随机采样
        sample_indices = np.random.choice(len(y_test_original), max_points, replace=False)
        sampled_y_test = y_test_original[sample_indices]
        sampled_y_pred = y_pred[sample_indices]
        print(f"数据量过大，已随机采样 {max_points} 个数据点进行可视化")
    else:
        sampled_y_test = y_test_original
        sampled_y_pred = y_pred

    plt.figure(figsize=(15, 8))
    plt.plot(sampled_y_test, label='实际值', c='blue', alpha=0.7)
    plt.plot(sampled_y_pred, label='预测值', c='red', alpha=0.7)
    plt.fill_between(range(len(sampled_y_test)), sampled_y_test, sampled_y_pred,
                     color='gray', alpha=0.2, label='误差区域')
    plt.xlabel('样本索引（采样）')
    plt.ylabel('值')
    plt.title('预测结果与实际值对比（采样版本）')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_plots/prediction_vs_actual_sampled.png', dpi=200, bbox_inches='tight')
    plt.close()

    if len(y_test_original) > 10000:
        chunk_size = 10000
        n_chunks = int(np.ceil(len(y_test_original) / chunk_size))

        print(f"数据量过大，将分 {n_chunks} 块进行可视化")

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(y_test_original))

            plt.figure(figsize=(15, 8))
            plt.plot(y_test_original[start_idx:end_idx], label='实际值', c='blue', alpha=0.7)
            plt.plot(y_pred[start_idx:end_idx], label='预测值', c='red', alpha=0.7)
            plt.fill_between(range(end_idx - start_idx),
                             y_test_original[start_idx:end_idx],
                             y_pred[start_idx:end_idx],
                             color='gray', alpha=0.2, label='误差区域')
            plt.xlabel(f'样本索引（块 {i + 1}/{n_chunks}）')
            plt.ylabel('值')
            plt.title(f'预测结果与实际值对比（块 {i + 1}/{n_chunks}）')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/prediction_plots/prediction_vs_actual_chunk_{i + 1}.png', dpi=200,
                        bbox_inches='tight')
            plt.close()

        plt.figure(figsize=(15, 8))

        group_size = min(1000, len(y_test_original) // 10)
        n_groups = int(np.ceil(len(y_test_original) / group_size))

        group_indices = np.arange(n_groups) * group_size
        test_means = np.array([np.mean(y_test_original[i:i + group_size])
                               for i in range(0, len(y_test_original), group_size)])
        pred_means = np.array([np.mean(y_pred[i:i + group_size])
                               for i in range(0, len(y_pred), group_size)])

        plt.plot(group_indices, test_means, label='实际值（均值）', c='blue', alpha=0.7)
        plt.plot(group_indices, pred_means, label='预测值（均值）', c='red', alpha=0.7)

        test_stds = np.array([np.std(y_test_original[i:i + group_size])
                              for i in range(0, len(y_test_original), group_size)])
        pred_stds = np.array([np.std(y_pred[i:i + group_size])
                              for i in range(0, len(y_pred), group_size)])

        plt.fill_between(group_indices,
                         test_means - test_stds,
                         test_means + test_stds,
                         color='blue', alpha=0.1, label='实际值标准差')
        plt.fill_between(group_indices,
                         pred_means - pred_stds,
                         pred_means + pred_stds,
                         color='red', alpha=0.1, label='预测值标准差')

        plt.xlabel('样本索引（按组）')
        plt.ylabel('值（均值±标准差）')
        plt.title('预测结果与实际值对比（分组统计）')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/prediction_plots/prediction_vs_actual_statistical.png', dpi=200, bbox_inches='tight')
        plt.close()

        print(f"已生成多种预测结果可视化图表，保存至 {output_dir}/prediction_plots/")
    else:
        try:
            plt.figure(figsize=(15, 8))
            plt.plot(y_test_original, label='实际值', c='blue', alpha=0.7)
            plt.plot(y_pred, label='预测值', c='red', alpha=0.7)
            plt.fill_between(range(len(y_test_original)), y_test_original, y_pred,
                             color='gray', alpha=0.2, label='误差区域')
            plt.xlabel('样本索引')
            plt.ylabel('值')
            plt.title('预测结果与实际值对比')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/prediction_plots/prediction_vs_actual.png', dpi=200, bbox_inches='tight')
            plt.close()
            print(f"已生成预测结果与实际值对比图，保存至 {output_dir}/prediction_plots/prediction_vs_actual.png")
        except Exception as e:
            print(f"生成完整预测结果图时发生错误: {e}")
            print(f"将尝试使用采样版本替代...")

            plt.figure(figsize=(15, 8))
            plt.plot(sampled_y_test, label='实际值', c='blue', alpha=0.7)
            plt.plot(sampled_y_pred, label='预测值', c='red', alpha=0.7)
            plt.fill_between(range(len(sampled_y_test)), sampled_y_test, sampled_y_pred,
                             color='gray', alpha=0.2, label='误差区域')
            plt.xlabel('样本索引（采样）')
            plt.ylabel('值')
            plt.title('预测结果与实际值对比（采样版本）')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/prediction_plots/prediction_vs_actual_sampled.png', dpi=200, bbox_inches='tight')
            plt.close()
            print(f"已使用采样版本生成预测结果图，保存至 {output_dir}/prediction_plots/prediction_vs_actual_sampled.png")

    print("\n训练与评估流程已全部完成！")
