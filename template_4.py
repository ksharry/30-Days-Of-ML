# 模版四：深度學習 (Deep Learning)
# 適用日子：Day 23-27 (MLP, CNN, RNN)
# ==========================================
# Day XX: [填入演算法名稱，如 Neural Network MLP]
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 模型與評估工具
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. 載入資料 (Data Loading) ---
# 設定資料集路徑與下載網址 (請依實際需求修改)
DATA_FILE = 'dataset.csv' 
DATA_URL = 'https://raw.githubusercontent.com/ksharry/30-Days-Of-ML/main/dayXX/dataset.csv'

def load_or_download_data(local_path, url):
    """
    檢查本地是否有檔案，若無則從 URL 下載。
    """
    if not os.path.exists(local_path):
        print(f"找不到檔案：{local_path}，嘗試從網路下載...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"下載成功！已儲存為 {local_path}")
        except Exception as e:
            print(f"下載失敗：{e}")
            return None
    else:
        print(f"發現本地檔案：{local_path}")
    
    return pd.read_csv(local_path)

# 嘗試讀取資料
try:
    df = load_or_download_data(DATA_FILE, DATA_URL)
    if df is None:
        raise ValueError("無法讀取資料")
except Exception as e:
    print(f"讀取自定義資料失敗 ({e})，改用 Scikit-Learn 內建糖尿病資料集範例 (回歸)...")
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Target'] = data.target

print(f"資料集維度: {df.shape}")
print(df.head())

# --- 2. 數據觀察 (EDA) ---
# 略，可參考其他模版

# --- 3. 資料分割 & 前處理 (DL 必做標準化) ---
X = df.drop('Target', axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 4. 建立與訓練模型 (Keras) ---
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), # 輸入層
    Dropout(0.2),
    Dense(32, activation='relu'), # 隱藏層
    Dense(1) # 輸出層 (回歸問題通常只有 1 個神經元，不加 activation)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 訓練並紀錄過程 (History)
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32,
                    validation_split=0.2, 
                    verbose=0) # verbose=0 不顯示大量 log

# --- 5. 模型評估 ---
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MSE: {loss:.4f}")
print(f"Test MAE: {mae:.4f}")

y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# --- 6. 解析權重 ---
model.summary() # 深度學習通常看結構摘要

# --- 7. 結果視覺化 (Training History) ---
# 這是深度學習最經典的圖：Loss 隨 Epoch 下降的曲線
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Prediction vs True Value')

plt.tight_layout()
plt.show()