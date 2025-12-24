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

# 視覺化 3: 神經網路架構示意圖 (Simplified Architecture)
def draw_neural_net_simplified(ax, left, right, bottom, top, layer_sizes, real_sizes):
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            
            if m == 0: text = "1"
            elif m == layer_size - 1: text = str(real_sizes[n])
            elif m == layer_size // 2: text = "..."
            else: text = ""
            
            if text:
                ax.text(n*h_spacing + left, layer_top - m*v_spacing, text, ha='center', va='center', fontsize=8)
        
        # Label the layer (Adjusted position)
        if n == 0: layer_name = "Input\n(Flatten)\n784"
        elif n == 1: layer_name = "Hidden\n(ReLU)\n128"
        else: layer_name = "Output\n(Softmax)\n10"
        # Move text slightly higher than the top node
        ax.text(n*h_spacing + left, top + 0.08, layer_name, ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                if m % 2 == 0 and o % 2 == 0:
                    line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                      [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k', alpha=0.1)
                    ax.add_artist(line)

# Change figsize to be wider and shorter to reduce bottom whitespace
fig = plt.figure(figsize=(12, 6))
ax = fig.gca()
ax.axis('off')
# Top is set to 0.8 to leave room for titles
draw_neural_net_simplified(ax, .1, .9, .1, .8, [10, 8, 10], [784, 128, 10])
plt.title('MLP Architecture for MNIST (Simplified View)', y=1.05) # Move title up
plt.savefig(os.path.join(pic_dir, '24-3_Network_Architecture.png'))
print("Network Architecture plot saved.")

# 視覺化 4: 梯度下降示意圖 (Gradient Descent Visualization)
x = np.linspace(-6, 6, 100)
y = x**2

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='Loss Function (Error)')

# Simulate Gradient Descent
current_x = -5
learning_rate = 0.2
path_x = [current_x]
path_y = [current_x**2]

for _ in range(5):
    gradient = 2 * current_x # derivative of x^2 is 2x
    current_x = current_x - learning_rate * gradient
    path_x.append(current_x)
    path_y.append(current_x**2)

plt.scatter(path_x, path_y, c='r', s=100, zorder=5, label='Steps')
for i in range(len(path_x)-1):
    plt.annotate('', xy=(path_x[i+1], path_y[i+1]), xytext=(path_x[i], path_y[i]),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))

plt.title('Gradient Descent: Rolling down the hill')
plt.xlabel('Weight (Parameter)')
plt.ylabel('Loss (Error)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(pic_dir, '24-4_Gradient_Descent.png'))
print("Gradient Descent plot saved.")
