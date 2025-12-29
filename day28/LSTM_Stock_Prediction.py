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

# 視覺化 2: LSTM 核心概念 (傳送帶與閘門)
def plot_lstm_concept():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    
    # Cell State (The Conveyor Belt) - Top Line
    ax.arrow(1, 5, 8, 0, head_width=0.2, head_length=0.2, fc='black', ec='black', width=0.05)
    ax.text(5, 5.3, "Cell State (Long-term Memory) $C_t$", ha='center', fontsize=12, fontweight='bold', color='blue')
    
    # Hidden State (Output) - Bottom Line
    ax.arrow(1, 1, 8, 0, head_width=0.2, head_length=0.2, fc='black', ec='black', width=0.05)
    ax.text(5, 0.5, "Hidden State (Short-term Memory) $h_t$", ha='center', fontsize=12, fontweight='bold', color='green')

    # The Gates (Rectangles)
    # Forget Gate
    rect_f = plt.Rectangle((2, 2.5), 1, 1, fc='#FF9999', ec='black')
    ax.add_patch(rect_f)
    ax.text(2.5, 3, "Forget\nGate\n($f_t$)", ha='center', va='center', fontweight='bold')
    ax.text(2.5, 2.2, "X", ha='center', va='center', fontsize=14, color='red', fontweight='bold') # Multiply
    
    # Input Gate
    rect_i = plt.Rectangle((4.5, 2.5), 1, 1, fc='#99FF99', ec='black')
    ax.add_patch(rect_i)
    ax.text(5, 3, "Input\nGate\n($i_t$)", ha='center', va='center', fontweight='bold')
    ax.text(5, 2.2, "+", ha='center', va='center', fontsize=14, color='green', fontweight='bold') # Add
    
    # Output Gate
    rect_o = plt.Rectangle((7, 2.5), 1, 1, fc='#99CCFF', ec='black')
    ax.add_patch(rect_o)
    ax.text(7.5, 3, "Output\nGate\n($o_t$)", ha='center', va='center', fontweight='bold')

    # Connections with Gaps
    arrow_params = dict(head_width=0.1, length_includes_head=True)

    # Inputs coming from bottom (Previous Hidden + Current Input)
    # End at 2.46 (Gap from 2.5)
    ax.arrow(2.5, 1.2, 0, 1.26, fc='gray', ec='gray', linestyle='--', **arrow_params)
    ax.arrow(5, 1.2, 0, 1.26, fc='gray', ec='gray', linestyle='--', **arrow_params)
    ax.arrow(7.5, 1.2, 0, 1.26, fc='gray', ec='gray', linestyle='--', **arrow_params)
    
    # Gates affecting Cell State
    # Start at 3.54 (Gap from 3.5)
    ax.arrow(2.5, 3.54, 0, 1.26, fc='red', ec='red', **arrow_params) # Forget
    ax.arrow(5, 3.54, 0, 1.26, fc='green', ec='green', **arrow_params) # Input
    
    # Cell State affecting Output
    # End at 3.54 (Gap from 3.5)
    ax.arrow(7.5, 4.8, 0, -1.26, fc='blue', ec='blue', **arrow_params) # From Cell to Output Gate
    
    # To Hidden State
    # Start at 2.46 (Gap from 2.5)
    ax.arrow(7.5, 2.46, 0, -1.26, fc='black', ec='black', **arrow_params) # To Hidden State

    plt.title("LSTM Internals: The Conveyor Belt & Three Gates", fontsize=14)
    plt.savefig(os.path.join(pic_dir, '28-2_LSTM_Concept.png'))
    print("LSTM Concept plot saved.")

plot_lstm_concept()
