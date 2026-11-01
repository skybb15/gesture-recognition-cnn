import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

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

# 提取标签和特征
y = all_data['label'].values
X = all_data.drop(['label'], axis=1).values

# 恢复图像形状
video_res_height_scaled = 100
video_res_width_scaled = X.shape[1] // video_res_height_scaled
X = X.reshape(-1, video_res_height_scaled, video_res_width_scaled, 1)

# 将标签转换为独热编码
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建 CNN 模型
model = models.Sequential([
    # 第一组卷积层 + 池化层
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(video_res_height_scaled, video_res_width_scaled, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # 第二组卷积层 + 池化层
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # 第三组卷积层 + 池化层
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # 展平层 + 全连接层
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # 防止过拟合
    layers.Dense(num_classes, activation='softmax')  # 输出层
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 打印模型结构
model.summary()

# 训练模型
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 保存模型
model.save('gesture_recognition_model.h5')

# 预测测试集
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 打印分类报告
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))

# 混淆矩阵
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# 绘制训练和验证的准确率曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Curve')

# 绘制训练和验证的损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curve')

plt.show()
