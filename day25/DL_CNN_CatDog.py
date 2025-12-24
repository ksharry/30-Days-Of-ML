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
def plot_convolution_diagram():
    fig, ax = plt.subplots(figsize=(10, 5)) # Shorter height
    ax.axis('off')
    
    # Draw Input Matrix (5x5)
    for i in range(5):
        for j in range(5):
            rect = plt.Rectangle((j, 4-i), 1, 1, fc='white', ec='black')
            ax.add_patch(rect)
            ax.text(j+0.5, 4-i+0.5, str(np.random.randint(0, 2)), ha='center', va='center')
    ax.text(2.5, 5.2, "Input Image (5x5)", ha='center', fontsize=12, fontweight='bold') # Moved up

    # Draw Filter (3x3) - Highlighted
    for i in range(3):
        for j in range(3):
            rect = plt.Rectangle((j, 4-i), 1, 1, fc='yellow', alpha=0.3, ec='red', lw=2)
            ax.add_patch(rect)
    ax.text(1.5, 6.0, "Filter (3x3)", ha='center', color='red', fontsize=10) # Moved up
    
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
    ax.text(9.5, 4.2, "Feature Map (3x3)", ha='center', fontsize=12, fontweight='bold') # Moved up
    
    plt.title("How Convolution Works: Sliding Window", y=1.1) # Moved title up
    plt.savefig(os.path.join(pic_dir, '25-3_Convolution_Diagram.png'))
    print("Convolution Diagram plot saved.")

plot_convolution_diagram()

# 視覺化 4: 池化運算示意圖 (Pooling Visualization)
def plot_pooling_diagram():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    # Input 4x4
    data = np.array([[1, 3, 2, 4],
                     [5, 6, 1, 2],
                     [8, 7, 3, 0],
                     [2, 1, 5, 4]])
    
    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99'] # Red, Green, Blue, Yellow
    
    # Draw Input
    for i in range(4):
        for j in range(4):
            # Determine color block
            block_idx = (i // 2) * 2 + (j // 2)
            rect = plt.Rectangle((j, 3-i), 1, 1, fc=colors[block_idx], alpha=0.3, ec='black')
            ax.add_patch(rect)
            ax.text(j+0.5, 3-i+0.5, str(data[i, j]), ha='center', va='center')
    ax.text(2, 4.2, "Input (4x4)", ha='center', fontsize=12, fontweight='bold')

    # Arrow
    ax.arrow(4.5, 2, 2, 0, head_width=0.2, head_length=0.3, fc='k', ec='k')
    ax.text(5.5, 2.5, "Max Pooling (2x2)", ha='center')

    # Draw Output 2x2
    output = np.array([[6, 4],
                       [8, 5]])
    
    for i in range(2):
        for j in range(2):
            block_idx = i * 2 + j
            rect = plt.Rectangle((7+j, 2.5-i), 1, 1, fc=colors[block_idx], alpha=0.5, ec='black')
            ax.add_patch(rect)
            ax.text(7+j+0.5, 2.5-i+0.5, str(output[i, j]), ha='center', va='center', fontweight='bold')
    ax.text(8, 3.7, "Output (2x2)", ha='center', fontsize=12, fontweight='bold')
    
    plt.title("How Max Pooling Works: Pick the Winner", y=1.1)
    plt.savefig(os.path.join(pic_dir, '25-4_Pooling_Diagram.png'))
    print("Pooling Diagram plot saved.")

plot_pooling_diagram()

# 視覺化 5: CNN 架構圖 (Architecture Visualization)
def draw_cnn_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    layers = [
        ("Input", "32x32", 32, 'white'),
        ("Conv1", "30x30", 30, '#FFCC99'), # Orange
        ("Pool1", "15x15", 15, '#99CCFF'), # Blue
        ("Conv2", "13x13", 13, '#FFCC99'),
        ("Pool2", "6x6", 6, '#99CCFF'),
        ("Flatten", "1D", 2, 'lightgray'),
        ("Dense", "Class", 1, 'lightgreen')
    ]
    
    x_pos = 0
    spacing = 2.5
    
    for i, (name, dim, size, color) in enumerate(layers):
        # Draw box
        height = size / 5.0 # Scale height
        width = 1.5
        if name == "Flatten": height = 6 # Represent long vector
        if name == "Dense": height = 1
        
        rect = plt.Rectangle((x_pos, 3 - height/2), width, height, fc=color, ec='black')
        ax.add_patch(rect)
        
        # Label
        ax.text(x_pos + width/2, 3 - height/2 - 0.5, name, ha='center', va='top', fontweight='bold')
        ax.text(x_pos + width/2, 3 + height/2 + 0.2, dim, ha='center', va='bottom', fontsize=9)
        
        # Arrow to next
        if i < len(layers) - 1:
            ax.arrow(x_pos + width + 0.2, 3, 0.6, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
        
        x_pos += spacing

    plt.title("CNN Architecture: C-P-C-P-F-D", y=1.05)
    plt.savefig(os.path.join(pic_dir, '25-5_CNN_Architecture.png'))
    print("CNN Architecture plot saved.")

draw_cnn_architecture()


