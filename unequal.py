import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import random
# 生成样本数据
with open('people.txt', 'r') as f:
    lines = f.readlines()

# 将出生和死亡日期提取为单独的数字，并存储在Numpy数组中
dates = []
count=0
for line in lines:
    name, birthdate, deathdate = line.strip().split(' ')
    birth_year = int(birthdate.replace("-", ""))
    death_year = int(deathdate.replace("-", ""))
    dates.append([birth_year, death_year])

pct_to_reverse = 0.05  # 需要颠倒的比例
num_to_reverse = int(len(dates) * pct_to_reverse)  # 需要颠倒的子列表数量
to_reverse = random.sample(dates, num_to_reverse)  # 随机选择需要颠倒的子列表

# 在选定的子列表中随机颠倒顺序
for sublist in to_reverse:
    if random.random() < 0.9:
        sublist.reverse()

dates = np.array(dates)
x=dates
y = np.zeros((x.shape[0], 1))
y[x[:, 0] < x[:, 1]] = 1
count=0
for i in y:
    if i == 0:
        count+=1
print(count)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 构建模型
model = Sequential([
    Dense(32, input_shape=(2,), activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

# 预测测试集数据
y_pred = model.predict(x_test)

# 将概率转换为二元分类结果
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 计算准确率、精确率和召回率
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
