# SVM 是一种监督学习算法，常用于分类和回归任务
from sklearn import svm


x = [[2, 0], [1, 1], [2, 3]]
y = [0, 0, 1]
clf = svm.SVC(kernel = 'linear')
# fit 方法用于训练模型，它接收特征数据 x 和目标标签 y 作为输入。
clf.fit(x, y)

print(clf); 

# get support vectors
print(clf.support_vectors_); # [[1. 1.] [2. 3.]]
# get indices of support vectors
print(clf.support_); # 打印支持向量的索引。
# get number of support vectors for each class
print(clf.n_support_); # 打印每个类别的支持向量数量。

# 新数据点
new_data = [[1, 2], [3, 4]]

# 预测新数据点的类别
predictions = clf.predict(new_data)

print("Predictions:", predictions)