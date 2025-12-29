# Day 27: RNN (Recurrent Neural Network) - 時間序列預測
# ---------------------------------------------------------
# 前面的 CNN 是處理「空間」資料 (圖片)。
# 今天的 RNN 是處理「時間」資料 (股票、語音、文字)。
# RNN 的特色是它有「記憶」，能記住前面的資訊來預測後面。
# 我們將產生模擬的股價資料 (正弦波 + 趨勢)，用 RNN 來預測未來的走勢。
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

print("Generating Simulated Stock Data...")

# 產生 1000 天的模擬股價 (Sine Wave + Trend + Noise)
t = np.arange(0, 1000)
# 股價 = sin(t) * 振幅 + 趨勢 + 雜訊
data = np.sin(0.02 * t) + 0.005 * t + np.random.normal(0, 0.1, 1000)

# 視覺化原始資料
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title("Simulated Stock Price (Sine + Trend + Noise)")
plt.xlabel("Day")
plt.ylabel("Price")
plt.savefig(os.path.join(pic_dir, '27-1_Raw_Data.png'))
print("Raw Data plot saved.")

# 資料前處理：製作「考古題」
# 我們要用「過去 10 天」的價格，來預測「明天」的價格
# X (Input): [Day 1, Day 2, ..., Day 10]
# y (Target): [Day 11]

def create_dataset(dataset, look_back=10):
    X, y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        X.append(a)
        y.append(dataset[i + look_back])
    return np.array(X), np.array(y)

look_back = 20 # 回看 20 天
X, y = create_dataset(data, look_back)

# 切分訓練集與測試集 (前 800 天訓練，後 200 天測試)
train_size = 800
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# RNN 的輸入格式要求：(Samples, Time Steps, Features)
# 這裡 Features = 1 (只有一個變數：價格)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(f"Train X shape: {X_train.shape}")
print(f"Train y shape: {y_train.shape}")

# --- 2. 建立模型 (Build RNN Model) ---
model = Sequential([
    # SimpleRNN 層
    # units=32: 32 個 RNN 神經元 (記憶單元)
    # input_shape=(look_back, 1): 輸入是 (20 天, 1 個價格)
    # activation='tanh': RNN 預設使用 tanh 激活函數
    SimpleRNN(32, input_shape=(look_back, 1), activation='tanh'),
    
    # 輸出層
    Dense(1) # 預測一個數值 (明天的價格)
])

model.compile(optimizer='adam', loss='mse') # 回歸問題常用 MSE (Mean Squared Error)
model.summary()

# --- 3. 訓練模型 (Training) ---
print("\nTraining RNN...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
print("Training complete.")

# --- 4. 模型評估與視覺化 ---
# 預測
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 視覺化結果
plt.figure(figsize=(14, 6))

# 1. 畫出真實資料
plt.plot(t, data, label='True Price', color='lightgray')

# 2. 畫出訓練集預測
# 訓練集的 x 軸範圍：從 look_back 開始，長度為 len(train_predict)
train_t = t[look_back : look_back + len(train_predict)]
plt.plot(train_t, train_predict, label='Train Predict', color='green')

# 3. 畫出測試集預測
# 測試集的 x 軸範圍：接在訓練集後面
test_start_t = look_back + len(train_predict)
test_t = t[test_start_t : test_start_t + len(test_predict)]
plt.plot(test_t, test_predict, label='Test Predict', color='red')

plt.title(f'RNN Stock Prediction (Look back {look_back} days)')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.savefig(os.path.join(pic_dir, '27-2_Prediction_Result.png'))
print("Prediction Result plot saved.")

# 視覺化 3: RNN 概念圖 (Unrolled RNN)
def plot_rnn_concept():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    ax.set_xlim(-1, 8)
    ax.set_ylim(-0.5, 4.5)
    
    # Draw Time Steps
    for i in range(3):
        x_offset = i * 3
        
        # Input (X)
        circle_x = plt.Circle((x_offset, 0), 0.4, fc='#AEC6CF', ec='black', lw=2)
        ax.add_patch(circle_x)
        ax.text(x_offset, 0, f"Input\n$X_{i}$", ha='center', va='center', fontweight='bold')
        
        # Hidden State (H) - The Memory
        rect_h = plt.Rectangle((x_offset - 0.5, 1.5), 1.0, 1.0, fc='#77DD77', ec='black', lw=2)
        ax.add_patch(rect_h)
        ax.text(x_offset, 2.0, f"Memory\n$H_{i}$", ha='center', va='center', fontweight='bold')
        
        # Output (Y)
        circle_y = plt.Circle((x_offset, 4.0), 0.4, fc='#FF6961', ec='black', lw=2)
        ax.add_patch(circle_y)
        ax.text(x_offset, 4.0, f"Output\n$Y_{i}$", ha='center', va='center', fontweight='bold')
        
        # Arrows with Weights
        # Use length_includes_head=True to make 'dx, dy' the exact tip of the arrow
        arrow_params = dict(head_width=0.15, head_length=0.15, fc='black', ec='black', length_includes_head=True)
        
        # X -> H (Weight W)
        # Start: 0.42 (Gap), End: 1.48 (Gap) -> Length = 1.06
        ax.arrow(x_offset, 0.42, 0, 1.06, **arrow_params, width=0.02)
        ax.text(x_offset - 0.2, 1.0, "$W$", ha='right', va='center', fontsize=12, color='blue', fontweight='bold')

        # H -> Y (Weight V)
        # Start: 2.52 (Gap), End: 3.58 (Gap) -> Length = 1.06
        ax.arrow(x_offset, 2.52, 0, 1.06, **arrow_params, width=0.02)
        ax.text(x_offset - 0.2, 3.0, "$V$", ha='right', va='center', fontsize=12, color='red', fontweight='bold')
        
        # Recurrent Arrow (H_i -> H_i+1) (Weight U)
        if i < 2:
            # Start: x+0.52 (Gap), End: x+2.48 (Gap) -> Length = 1.96
            ax.arrow(x_offset + 0.52, 2.0, 1.96, 0, **arrow_params, width=0.02)
            ax.text(x_offset + 1.5, 2.2, "$U$\n(Memory)", ha='center', va='bottom', fontsize=12, color='green', fontweight='bold')

    plt.title("RNN Unrolled: Weights (W, U, V) and Memory Flow", y=1.05, fontsize=14)
    plt.savefig(os.path.join(pic_dir, '27-3_RNN_Concept.png'))
    print("RNN Concept plot saved.")

plot_rnn_concept()
