# Day 25: CNN (Convolutional Neural Network) - 貓狗圖片分類
# ---------------------------------------------------------
# 這一天的目標是學習影像辨識的王牌：CNN (卷積神經網路)。
# 昨天我們用 MLP 辨識數字，但 MLP 有個大問題：它把圖片壓扁了！
# CNN 透過「卷積 (Convolution)」和「池化 (Pooling)」來保留空間特徵。
# 我們將使用一個簡化版的貓狗資料集 (CIFAR-10 的貓狗子集) 來示範。
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

print("Loading CIFAR-10 Dataset...")
# CIFAR-10 包含 10 類彩色圖片 (32x32x3)，其中 Class 3 是貓，Class 5 是狗
(X_train_full, y_train_full), (X_test_full, y_test_full) = cifar10.load_data()

# 只挑選「貓 (3)」和「狗 (5)」的資料
def filter_cat_dog(X, y):
    keep = (y.flatten() == 3) | (y.flatten() == 5)
    X_keep = X[keep]
    y_keep = y[keep]
    # 將標籤改為 0 (貓) 和 1 (狗)
    y_keep = np.where(y_keep == 3, 0, 1)
    return X_keep, y_keep

X_train, y_train = filter_cat_dog(X_train_full, y_train_full)
X_test, y_test = filter_cat_dog(X_test_full, y_test_full)

print(f"Cat/Dog Train shape: {X_train.shape}")
print(f"Cat/Dog Test shape: {X_test.shape}")

# 正規化 (Normalization)
X_train = X_train / 255.0
X_test = X_test / 255.0

# --- 2. 建立模型 (Build CNN Model) ---
model = Sequential([
    # 第一層卷積：32 個 3x3 的濾鏡 (Filter)，負責抓取邊緣、線條等特徵
    # input_shape=(32, 32, 3): 32x32 像素，3 個顏色通道 (RGB)
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name='Conv_1'),
    
    # 第一層池化：2x2 的最大池化 (Max Pooling)，負責縮小圖片、保留最明顯特徵
    MaxPooling2D((2, 2), name='Pool_1'),
    
    # 第二層卷積：64 個 3x3 的濾鏡，負責抓取更複雜的形狀 (如眼睛、耳朵)
    Conv2D(64, (3, 3), activation='relu', name='Conv_2'),
    
    # 第二層池化
    MaxPooling2D((2, 2), name='Pool_2'),
    
    # 攤平層：將 2D 特徵圖拉平成 1D 向量，準備餵給全連接層
    Flatten(name='Flatten'),
    
    # 全連接層 (Dense)：負責最後的分類判斷
    Dense(64, activation='relu', name='Dense_Hidden'),
    
    # 輸出層：1 個神經元 (Sigmoid)，輸出是狗的機率 (0~1)
    Dense(1, activation='sigmoid', name='Output')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 3. 訓練模型 (Training) ---
print("\nTraining CNN...")
# epochs=10: 訓練 10 輪
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
print("Training complete.")

# --- 4. 模型評估與視覺化 ---
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# 視覺化 1: 訓練過程
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss History')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy History')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(pic_dir, '25-1_Training_History.png'))
print("Training History plot saved.")

# 視覺化 2: 預測結果展示
indices = np.random.choice(len(X_test), 15, replace=False)
images = X_test[indices]
true_labels = y_test[indices]
predictions = model.predict(images)
# 因為是 Sigmoid，大於 0.5 判斷為 1 (狗)，否則為 0 (貓)
predicted_classes = (predictions > 0.5).astype(int).flatten()

class_names = ['Cat', 'Dog']

plt.figure(figsize=(15, 6))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(images[i])
    
    color = 'green' if predicted_classes[i] == true_labels[i] else 'red'
    label_text = f"Pred: {class_names[predicted_classes[i]]}\nTrue: {class_names[true_labels[i][0]]}"
    
    plt.title(label_text, color=color)
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '25-2_Predictions.png'))
print("Predictions plot saved.")

# 視覺化 3: 卷積運算示意圖 (Convolution Visualization)
# 畫一個簡單的 5x5 矩陣和 3x3 濾鏡，展示卷積過程
def plot_convolution_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Draw Input Matrix (5x5)
    for i in range(5):
        for j in range(5):
            rect = plt.Rectangle((j, 4-i), 1, 1, fc='white', ec='black')
            ax.add_patch(rect)
            ax.text(j+0.5, 4-i+0.5, str(np.random.randint(0, 2)), ha='center', va='center')
    ax.text(2.5, 5.5, "Input Image (5x5)", ha='center', fontsize=12, fontweight='bold')

    # Draw Filter (3x3) - Highlighted
    for i in range(3):
        for j in range(3):
            rect = plt.Rectangle((j, 4-i), 1, 1, fc='yellow', alpha=0.3, ec='red', lw=2)
            ax.add_patch(rect)
    ax.text(1.5, 6.5, "Filter (3x3)", ha='center', color='red', fontsize=10)
    
    # Draw Arrow
    ax.arrow(5.5, 2.5, 2, 0, head_width=0.2, head_length=0.3, fc='k', ec='k')
    ax.text(6.5, 3, "Convolution", ha='center')

    # Draw Output Feature Map (3x3)
    for i in range(3):
        for j in range(3):
            rect = plt.Rectangle((8+j, 3-i), 1, 1, fc='lightblue', ec='black')
            ax.add_patch(rect)
            if i==0 and j==0:
                ax.text(8+j+0.5, 3-i+0.5, "?", ha='center', va='center', fontweight='bold')
    ax.text(9.5, 4.5, "Feature Map (3x3)", ha='center', fontsize=12, fontweight='bold')
    
    plt.title("How Convolution Works: Sliding Window", y=1.05)
    plt.savefig(os.path.join(pic_dir, '25-3_Convolution_Diagram.png'))
    print("Convolution Diagram plot saved.")

plot_convolution_diagram()
