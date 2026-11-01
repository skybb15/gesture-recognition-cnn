import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 设置视频和CSV存储的路径
video_folder = './dataset/'
csv_folder = './dataset_csv/'

# 确保输出CSV文件夹存在
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# 图像参数
video_res_height_scaled = 100  # 缩放后的高度
num_frames = 50  # 每个视频提取的帧数

# 循环读取 gesture_0.mp4 到 gesture_9.mp4
for gesture_label in range(10):
    video_file = os.path.join(video_folder, f'gesture_{gesture_label}.mp4')
    save_file_name = os.path.join(csv_folder, f'gesture_{gesture_label}.csv')

    # 打开视频文件
    video_capture = cv2.VideoCapture(video_file)
    if not video_capture.isOpened():
        print(f"Error: Could not open {video_file}")
        continue

    # 获取视频的宽高比并计算调整后的宽度
    width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_res_ratio = width / height
    video_res_width_scaled = int(video_res_height_scaled * video_res_ratio)

    # 创建DataFrame存储数据
    df = pd.DataFrame()
    frame_count = 0

    while frame_count < num_frames:
        success, frame = video_capture.read()
        if not success:
            break
        frame_count += 1

        # 灰度转换与分辨率调整
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (video_res_width_scaled, video_res_height_scaled))

        # 提取像素值并存储到DataFrame
        pixels = gray_frame.reshape(-1)
        temp_df = pd.DataFrame([pixels], columns=[f'pixel{i + 1}' for i in range(len(pixels))])
        temp_df.insert(0, 'label', gesture_label)

        df = pd.concat([df, temp_df], ignore_index=True)

    # 保存数据到CSV
    df.to_csv(save_file_name, index=False)
    print(f"Saved: {save_file_name}")

    video_capture.release()

print("All videos processed successfully.")

# 数据验证与增强
# ====================================================
# 合并所有CSV文件
all_data = pd.DataFrame()
for gesture_label in range(10):
    file_path = os.path.join(csv_folder, f'gesture_{gesture_label}.csv')
    if os.path.exists(file_path):
        temp_data = pd.read_csv(file_path)
        all_data = pd.concat([all_data, temp_data], ignore_index=True)

# 提取标签和特征
y_train = all_data['label']
all_data.drop(['label'], axis=1, inplace=True)

X_train = all_data.values.reshape(all_data.shape[0], video_res_height_scaled, video_res_width_scaled, 1)

# 数据增强设置
train_datagen = ImageDataGenerator(
    rescale=1./255,                   # 归一化
    zoom_range=0.2,                   # 随机缩放
    width_shift_range=0.1,            # 随机水平平移
    height_shift_range=0.1,           # 随机垂直平移
    rotation_range=20,                # 随机旋转角度
    shear_range=0.2,                  # 随机剪切变换
    brightness_range=[0.7, 1.3],      # 随机亮度调节
    horizontal_flip=True,             # 随机水平翻转
    fill_mode='nearest'               # 填充策略
)

# 数据生成器
X_train_augmented = train_datagen.flow(X_train, y_train, batch_size=5, shuffle=True)

# 样本可视化
plt.figure(figsize=(25, 25))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    img, label = next(X_train_augmented)  # 获取增强数据
    plt.imshow(img[0].reshape(video_res_height_scaled, -1), cmap='gray')
    plt.title(f'Label: {label[0]}')
    plt.axis('off')
plt.show()

