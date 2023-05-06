from sklearn.ensemble import IsolationForest
import numpy as np

# 打开文件
with open('people.txt', 'r') as f:
    lines = f.readlines()

# 将出生和死亡日期提取为单独的数字，并存储在Numpy数组中
dates = []
for line in lines:
    name, birthdate, deathdate = line.strip().split(' ')
    birth_year = int(birthdate.replace("-", ""))
    death_year = int(deathdate.replace("-", ""))
    dates.append([birth_year, death_year])


# 打印Numpy数组
#print(dates)

# 随机生成1000个10维向量作为数据集
# X = np.random.rand(1000, 10)
X = np.array(dates)

# 将数据集分为训练集和测试集
train_ratio = 0.8
train_size = int(train_ratio * X.shape[0])
X_train, X_test = X[:train_size], X[train_size:]


# 训练Isolation Forest模型
clf = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', random_state=0)
clf.fit(X_train)

# 预测测试集中的离群点
y_pred = clf.predict(X_test)

# 输出预测结果
print("Predictions:", y_pred)

# 判断新数据点是否为离群点
new_point = np.random.rand(1, 2)
if clf.predict(new_point)[0] == -1:
    print("The new point is an outlier.")
else:
    print("The new point is not an outlier.")
