import numpy as np # 用于数值计算
import pylab as pl # 用于绘图
from sklearn import svm # 是 scikit-learn 中的支持向量机模块

# 创建两个类别的数据点，每个类别有 20 个样本。第一类数据点的均值为 [-2, -2]，第二类数据点的均值为 [+2, +2]。
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0]*20 +[1]*20

#fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# 获取分离超平面
w = clf.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-5, 5)
yy = a*xx - (clf.intercept_[0])/w[1]

# 绘制支持向量和平行线
b = clf.support_vectors_[0]
yy_down = a*xx + (b[1] - a*b[0])
b = clf.support_vectors_[-1]
yy_up = a*xx + (b[1] - a*b[0])

print("w: ", w);
print("a: ", a);

# print("xx: ", xx);
# print("yy: ", yy);
print("support_vectors_: ", clf.support_vectors_);
print("clf.coef_: ", clf.coef_);

# switching to the generic n-dimensional parameterization of the hyperplan to the 2D-specific equation
# of a line y=a.x +b: the generic w_0x + w_1y +w_3=0 can be rewritten y = -(w_0/w_1) x + (w_3/w_1)


# 绘制图形
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
          s=80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()