from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()

# 加载鸢尾花数据集

iris = datasets.load_iris()

# 保存数据
# f = open("iris.data.csv", 'wb')
# f.write(str(iris).encode('utf-8'))  # 将字符串编码为字节
# f.close()

# 加载鸢尾花数据集
print(iris)

# 使用鸢尾花数据集的特征和目标标签训练 KNN 模型。
knn.fit(iris.data, iris.target)
# 对一个新的样本 [6.3, 2.7, 4.9, 1.8] 进行预测，返回预测的类别标签。
predictedLabel = knn.predict([[6.3, 2.7, 4.9, 1.8]])
print("predictedLabel is:", predictedLabel) # [2]