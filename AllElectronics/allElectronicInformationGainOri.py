from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
import six
from six import StringIO

# Read in the csv file and put features into list of dict and list of class label
with open(r'D:\AllElectronics.csv', 'r', encoding='utf-8') as allElectronicsData:
    reader = csv.reader(allElectronicsData)
    headers = next(reader)  # 使用next函数获取表头

    print(headers)

    featureList = []
    labelList = []

    for row in reader:
        labelList.append(row[len(row)-1])
        rowDict = {}
        for i in range(1, len(row)-1):
            rowDict[headers[i]] = row[i]
        featureList.append(rowDict)

    print(featureList)

# Vectorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print("dummyX: " + str(dummyX))
print(vec.get_feature_names_out())  # 使用 get_feature_names_out()

print("labelList: " + str(labelList))

# Vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))

# Visualize model
with open(r'D:\allElectronicInformationGainOri.dot', 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names_out(), out_file=f)

oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

predictedY = clf.predict([newRowX])  # 注意这里需要传入一个二维数组
print("predictedY: " + str(predictedY))