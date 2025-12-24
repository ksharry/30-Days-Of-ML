# Day 23: Deep Learning Basics - Perceptron & MLP (XOR Problem)
# ---------------------------------------------------------
# 這一天的目標是進入深度學習 (Deep Learning) 的世界。
# 我們從最基本的單元：感知機 (Perceptron) 開始。
# 並挑戰一個經典難題：XOR 問題 (互斥或)。
# 單層感知機無法解決 XOR，必須使用多層感知機 (MLP)。
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os

# 嘗試匯入 TensorFlow，如果沒有安裝則提示
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("Error: TensorFlow is not installed.")
    print("Please install it using: pip install tensorflow")
    exit(1)

# --- 1. 準備資料 (XOR Problem) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# XOR 資料集：
# (0, 0) -> 0
# (0, 1) -> 1
# (1, 0) -> 1
# (1, 1) -> 0
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

print("XOR Data:")
print(X)
print("Labels:")
print(y)

# --- 2. 建立模型 (MLP - Multi-Layer Perceptron) ---
# 我們需要一個隱藏層 (Hidden Layer) 來解決非線性問題
# 結構：Input(2) -> Hidden(8, ReLU) -> Output(1, Sigmoid)
model = Sequential([
    # 隱藏層：8 個神經元，使用 ReLU 激活函數
    # input_dim=2: 因為輸入只有兩個特徵 (x1, x2)
    Dense(8, input_dim=2, activation='relu', name='Hidden_Layer'),
    
    # 輸出層：1 個神經元，使用 Sigmoid 激活函數 (輸出 0~1 的機率)
    Dense(1, activation='sigmoid', name='Output_Layer')
])

# 編譯模型
# optimizer='adam': 目前最強大的優化器
# loss='binary_crossentropy': 二元分類的標準損失函數
model.compile(optimizer=Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# --- 3. 訓練模型 ---
print("\nTraining MLP...")
# epochs=100: 訓練 100 輪
history = model.fit(X, y, epochs=100, verbose=0)
print("Training complete.")

# --- 4. 模型評估與視覺化 ---
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\nFinal Accuracy: {accuracy:.4f}")
print(f"Final Loss: {loss:.4f}")

# 預測結果
predictions = model.predict(X)
print("\nPredictions:")
print(predictions.round())

# 視覺化 1: 訓練過程 (Loss Curve)
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('Training History (MLP on XOR)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(pic_dir, '23-1_Training_History.png'))
print("Training History plot saved.")

# 視覺化 2: 決策邊界 (Decision Boundary)
# 為了畫出漂亮的邊界，我們產生密集的網格點
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid_points)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap='RdBu', alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=200, cmap='RdBu', edgecolors='white', linewidth=2)
# 標上座標點
for i in range(4):
    plt.text(X[i, 0]+0.05, X[i, 1]+0.05, f"({X[i,0]:.0f},{X[i,1]:.0f})", fontsize=12, fontweight='bold')

plt.title('Decision Boundary (MLP solves XOR)')
plt.xlabel('Input x1')
plt.ylabel('Input x2')
plt.savefig(os.path.join(pic_dir, '23-2_Decision_Boundary.png'))
print("Decision Boundary plot saved.")

# 視覺化 3: 神經網路架構圖 (Neural Network Architecture)
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            # Add text
            if n == 0: text = f"x{m+1}"
            elif n == len(layer_sizes)-1: text = "y"
            else: text = f"h{m+1}"
            ax.text(n*h_spacing + left, layer_top - m*v_spacing, text, ha='center', va='center', fontsize=8)

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k', alpha=0.2)
                ax.add_artist(line)

fig = plt.figure(figsize=(12, 6))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, .1, .9, .1, .8, [2, 8, 1])
plt.title('MLP Architecture (2 Inputs -> 8 Hidden -> 1 Output)', y=1.05)
plt.savefig(os.path.join(pic_dir, '23-3_Network_Architecture.png'))
print("Network Architecture plot saved.")
