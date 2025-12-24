# Day 24: Deep Learning Training - MNIST 手寫數字辨識
# ---------------------------------------------------------
# 這一天的目標是挑戰深度學習的 "Hello World"：MNIST。
# 我們將建立一個多層感知機 (MLP) 來辨識 0~9 的手寫數字。
# 重點在於理解神經網路的訓練過程：
# 1. Forward Propagation (前向傳播)：預測結果。
# 2. Loss Calculation (計算誤差)：跟答案差多少？
# 3. Backpropagation (反向傳播)：將誤差傳回去。
# 4. Optimizer (優化器)：更新權重 (Adam)。
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

print("Loading MNIST Dataset...")
# MNIST 包含 60,000 筆訓練資料，10,000 筆測試資料
# 圖片大小為 28x28 像素，灰階 (0~255)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")

# 資料前處理 (Preprocessing)
# 1. 正規化 (Normalization)：將像素值從 0~255 縮放到 0~1
#    這對神經網路的收斂非常有幫助
X_train = X_train / 255.0
X_test = X_test / 255.0

# --- 2. 建立模型 (Build Model) ---
model = Sequential([
    # 輸入層：將 28x28 的二維圖片「攤平」成 784 個一維向量
    Flatten(input_shape=(28, 28), name='Input_Flatten'),
    
    # 隱藏層：128 個神經元，使用 ReLU 激活函數
    Dense(128, activation='relu', name='Hidden_Layer'),
    
    # 輸出層：10 個神經元 (對應 0~9 十個數字)，使用 Softmax 輸出機率分佈
    Dense(10, activation='softmax', name='Output_Layer')
])

# 編譯模型
model.compile(
    optimizer='adam',  # 自適應學習率優化器
    loss='sparse_categorical_crossentropy', # 多分類問題的標準損失函數 (標籤為整數時用 sparse)
    metrics=['accuracy']
)

model.summary()

# --- 3. 訓練模型 (Training) ---
print("\nTraining Model...")
# batch_size=32: 每次看 32 張圖就調整一次參數
# validation_split=0.1: 從訓練資料切 10% 出來當驗證集 (考試前的小考)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
print("Training complete.")

# --- 4. 模型評估與視覺化 ---
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# 視覺化 1: 訓練過程 (Loss & Accuracy)
plt.figure(figsize=(12, 5))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '24-1_Training_History.png'))
print("Training History plot saved.")

# 視覺化 2: 預測結果展示 (Predictions)
# 隨機挑選 15 張測試圖片來預測
indices = np.random.choice(len(X_test), 15, replace=False)
images = X_test[indices]
true_labels = y_test[indices]
predictions = model.predict(images)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(15, 6))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    
    # 如果預測正確用綠色，錯誤用紅色
    color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
    
    plt.title(f"Pred: {predicted_labels[i]}\nTrue: {true_labels[i]}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '24-2_Predictions.png'))
print("Predictions plot saved.")
