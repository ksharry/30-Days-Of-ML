# Day 28: LSTM (Long Short-Term Memory) - 解決金魚腦
# ---------------------------------------------------------
# 昨天 RNN 的缺點是 "梯度消失" (記不住長期的事)。
# 今天我們用 LSTM，它有 "傳送帶 (Cell State)" 和 "三個門 (Gates)"。
# 這讓它能精準控制 "該忘記什麼" 和 "該記住什麼"。
# 我們將使用同樣的模擬股價資料，但這次嘗試更長的 Look-back。
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

print("Generating Simulated Stock Data...")

# 產生 1000 天的模擬股價 (Sine Wave + Trend + Noise)
# 與 Day 27 相同，以便比較
t = np.arange(0, 1000)
data = np.sin(0.02 * t) + 0.005 * t + np.random.normal(0, 0.1, 1000)

def create_dataset(dataset, look_back=10):
    X, y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        X.append(a)
        y.append(dataset[i + look_back])
    return np.array(X), np.array(y)

# 挑戰：增加 Look-back 天數，測試 LSTM 的記憶力
# RNN 在長序列容易忘記，LSTM 則比較穩
look_back = 50 
X, y = create_dataset(data, look_back)

# 切分訓練集與測試集
train_size = 800
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM 輸入格式: (Samples, Time Steps, Features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(f"Train X shape: {X_train.shape}")

# --- 2. 建立模型 (Build LSTM Model) ---
model = Sequential([
    # LSTM 層
    # units=50: 50 個 LSTM 單元 (比 RNN 複雜，通常需要多一點單元)
    LSTM(50, input_shape=(look_back, 1), activation='tanh'),
    
    # 輸出層
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# --- 3. 訓練模型 (Training) ---
print("\nTraining LSTM...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
print("Training complete.")

# --- 4. 模型評估與視覺化 ---
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 視覺化 1: 預測結果
plt.figure(figsize=(14, 6))

# 1. 真實資料
plt.plot(t, data, label='True Price', color='lightgray')

# 2. 訓練集預測
train_t = t[look_back : look_back + len(train_predict)]
plt.plot(train_t, train_predict, label='Train Predict (LSTM)', color='green')

# 3. 測試集預測
test_start_t = look_back + len(train_predict)
test_t = t[test_start_t : test_start_t + len(test_predict)]
plt.plot(test_t, test_predict, label='Test Predict (LSTM)', color='red')

plt.title(f'LSTM Stock Prediction (Look back {look_back} days)')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.savefig(os.path.join(pic_dir, '28-1_Prediction_Result.png'))
print("Prediction Result plot saved.")

# 視覺化 2: LSTM 核心概念 (Colah Style)
def plot_lstm_concept():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    
    # Styles
    layer_color = '#F7DC6F' # Yellow
    op_color = '#F1948A'    # Pink
    line_color = 'black'
    arrow_params = dict(head_width=0.2, head_length=0.2, fc=line_color, ec=line_color, length_includes_head=True)

    # --- 1. Top Rail (Cell State) ---
    # C_{t-1} -> X -> + -> C_t
    ax.text(0.5, 7.5, "$C_{t-1}$", fontsize=14, fontweight='bold', ha='center')
    ax.arrow(1, 7.5, 1.5, 0, **arrow_params) # To Multiply
    
    # Multiply Op (Forget)
    circle_mul1 = plt.Circle((3, 7.5), 0.4, fc=op_color, ec='black')
    ax.add_patch(circle_mul1)
    ax.text(3, 7.5, "$\\times$", fontsize=14, ha='center', va='center')
    
    ax.arrow(3.4, 7.5, 2.6, 0, **arrow_params) # To Add
    
    # Add Op (Update)
    circle_add = plt.Circle((6.5, 7.5), 0.4, fc=op_color, ec='black')
    ax.add_patch(circle_add)
    ax.text(6.5, 7.5, "$+$", fontsize=14, ha='center', va='center')
    
    ax.arrow(6.9, 7.5, 4, 0, **arrow_params) # To C_t
    ax.text(11.5, 7.5, "$C_t$", fontsize=14, fontweight='bold', ha='center')

    # --- 2. Bottom Inputs ---
    ax.text(0.5, 1.5, "$h_{t-1}$", fontsize=14, fontweight='bold', ha='center')
    ax.arrow(1, 1.5, 0.5, 0, **arrow_params) # h_{t-1} in
    
    ax.text(1.5, 0.5, "$x_t$", fontsize=14, fontweight='bold', ha='center')
    ax.arrow(1.5, 1, 0, 0.5, **arrow_params) # x_t in
    
    # Merge line going right
    ax.plot([1.5, 8.5], [1.5, 1.5], color=line_color) 

    # --- 3. Gates (Layers) ---
    
    # Forget Gate (Sigmoid)
    # Path: Bottom -> Sigmoid -> Top Multiply
    ax.arrow(3, 1.5, 0, 1.5, **arrow_params) # Up to layer
    rect_f = plt.Rectangle((2.5, 3), 1, 1, fc=layer_color, ec='black')
    ax.add_patch(rect_f)
    ax.text(3, 3.5, "$\\sigma$", fontsize=14, ha='center', va='center')
    ax.text(3, 2.2, "Forget", fontsize=10, ha='center') # Moved down
    ax.arrow(3, 4, 0, 3.1, **arrow_params) # Up to Multiply
    
    # Input Gate (Sigmoid)
    # Path: Bottom -> Sigmoid -> Multiply
    ax.arrow(4.5, 1.5, 0, 1.5, **arrow_params) # Up to layer
    rect_i = plt.Rectangle((4, 3), 1, 1, fc=layer_color, ec='black')
    ax.add_patch(rect_i)
    ax.text(4.5, 3.5, "$\\sigma$", fontsize=14, ha='center', va='center')
    ax.text(4.5, 2.2, "Input", fontsize=10, ha='center') # Moved down
    
    # Candidate Layer (Tanh)
    # Path: Bottom -> Tanh -> Multiply
    ax.arrow(5.5, 1.5, 0, 1.5, **arrow_params) # Up to layer
    rect_c = plt.Rectangle((5, 3), 1, 1, fc=layer_color, ec='black')
    ax.add_patch(rect_c)
    ax.text(5.5, 3.5, "$\\tanh$", fontsize=12, ha='center', va='center')
    ax.text(5.5, 2.2, "Cand.", fontsize=10, ha='center') # Moved down
    
    # Merge Input & Candidate
    ax.arrow(4.5, 4, 0, 1.1, **arrow_params) # From Input Gate
    ax.arrow(5.5, 4, 0, 1.1, **arrow_params) # From Candidate
    
    # Multiply Op (Input * Candidate)
    circle_mul2 = plt.Circle((5, 5.5), 0.4, fc=op_color, ec='black')
    ax.add_patch(circle_mul2)
    ax.text(5, 5.5, "$\\times$", fontsize=14, ha='center', va='center')
    
    # Connect Merge to Multiply
    ax.plot([4.5, 5.5], [5.1, 5.1], color='black') # Horizontal bar
    ax.arrow(5, 5.1, 0, 0.05, head_width=0, color='black') # Tiny connector
    
    # Result to Top Add
    ax.arrow(5, 5.9, 1.1, 1.2, **arrow_params) # To Add
    
    # Output Gate (Sigmoid)
    # Path: Bottom -> Sigmoid -> Output Multiply
    ax.arrow(8.5, 1.5, 0, 1.5, **arrow_params) # Up to layer
    rect_o = plt.Rectangle((8, 3), 1, 1, fc=layer_color, ec='black')
    ax.add_patch(rect_o)
    ax.text(8.5, 3.5, "$\\sigma$", fontsize=14, ha='center', va='center')
    ax.text(8.5, 2.2, "Output", fontsize=10, ha='center') # Moved down
    
    # Output Gate Arrow to Multiply (Adjusted path)
    # Go up, then right, then up to avoid overlap
    ax.plot([8.5, 8.5], [4, 4.5], color='black') # Up
    ax.arrow(8.5, 4.5, 0.6, 0, **arrow_params) # Right to Multiply
    
    # Output Tanh (on Cell State branch)
    # Branch from Top Rail
    ax.plot([9.5, 9.5], [7.5, 6.5], color='black') # Down from top
    
    # Tanh Op (Moved up)
    circle_tanh = plt.Circle((9.5, 6.0), 0.5, fc=op_color, ec='black')
    ax.add_patch(circle_tanh)
    ax.text(9.5, 6.0, "$\\tanh$", fontsize=10, ha='center', va='center')
    
    ax.arrow(9.5, 5.5, 0, -0.6, **arrow_params) # Down to Multiply (Longer arrow)
    
    # Output Multiply
    circle_mul3 = plt.Circle((9.5, 4.5), 0.4, fc=op_color, ec='black')
    ax.add_patch(circle_mul3)
    ax.text(9.5, 4.5, "$\\times$", fontsize=14, ha='center', va='center')
    
    # Final Output h_t
    ax.arrow(9.9, 4.5, 1.6, 0, **arrow_params)
    ax.text(12, 4.5, "$h_t$", fontsize=14, fontweight='bold', ha='center')

    plt.title("LSTM Structure (Standard View)", fontsize=16, y=1.02)
    plt.savefig(os.path.join(pic_dir, '28-2_LSTM_Concept.png'))
    print("LSTM Concept plot saved.")

plot_lstm_concept()
