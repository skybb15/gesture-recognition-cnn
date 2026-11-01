import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# 加载所有CSV文件并合并
file_paths = [
    './dataset_csv/gesture_0.csv', './dataset_csv/gesture_1.csv',
    './dataset_csv/gesture_2.csv', './dataset_csv/gesture_3.csv',
    './dataset_csv/gesture_4.csv', './dataset_csv/gesture_5.csv',
    './dataset_csv/gesture_6.csv', './dataset_csv/gesture_7.csv',
    './dataset_csv/gesture_8.csv', './dataset_csv/gesture_9.csv'
]

# 合并所有手势数据
all_data = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)

# 提取特征和标签
y = all_data['label'].values
X = all_data.drop(['label'], axis=1).values

# 恢复图像形状
video_res_height_scaled = 100
video_res_width_scaled = X.shape[1] // video_res_height_scaled
X = X.reshape(-1, video_res_height_scaled, video_res_width_scaled, 1)

# 将问题转化为二分类（例如，0 为正类，其他为负类）
y_binary = (y == 0).astype(int)  # 如果是类别 0，则为正类，否则为负类

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# 构建 CNN 模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(video_res_height_scaled, video_res_width_scaled, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # 输出为一个神经元，sigmoid 激活用于二分类
])

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # 二分类交叉熵损失
              metrics=['accuracy'])

# 打印模型结构
model.summary()

# 训练模型
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 保存模型
model.save('gesture_binary_classification_model.h5')

# 测试模型
y_pred_prob = model.predict(X_test)  # 获取预测概率
y_pred = (y_pred_prob > 0.5).astype(int)  # 转为二分类预测

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

# 计算性能指标
accuracy = (tp + tn) / (tp + tn + fp + fn)  # 准确率
sensitivity = tp / (tp + fn)  # 灵敏性（召回率）
specificity = tn / (tn + fp)  # 特异性

print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# 打印分类报告
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 计算 ROC 曲线和 AUC 值
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 基准线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

