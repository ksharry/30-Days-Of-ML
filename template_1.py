# 模版 1：通用回歸分析架構 (The Universal Regression Template)
# 適用日子：Day 02, 03, 04, 05 升級點：標準化數據、熱力圖、更嚴謹的資料切分。
# ==========================================
# Day 02-05: 通用回歸模版 (Regression Template)
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests

# 模型與評估工具
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 根據 Day 換模型：
from sklearn.linear_model import LinearRegression 
# from sklearn.linear_model import Lasso, Ridge (Day 04)

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
    print(f"讀取自定義資料失敗 ({e})，改用 Scikit-Learn 內建加州房價資料集範例...")
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Target'] = data.target

print(f"資料集維度: {df.shape}")
print(df.head())

# --- 2. 數據觀察 (EDA) ---
# [通用] 檢查相關係數矩陣 (Day 03 重點：檢查多重共線性)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# [選用] 觀察特定特徵與目標的關係
# sns.scatterplot(x='MedInc', y='Target', data=df, alpha=0.3)
# plt.show()

# --- 3. 資料分割與前處理 (關鍵升級) ---
X = df.drop('Target', axis=1)
y = df['Target']

# 切分資料 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# [Day 04/05 必備] 標準化 (Standardization)
# 讓所有特徵都在同一個起跑點，這樣係數 (Weights) 才有比較意義
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # 注意：測試集只能 transform，不能 fit！

# --- 4. 建立與訓練模型 ---
model = LinearRegression()
# model = Ridge(alpha=1.0) # Day 04 換這個
model.fit(X_train_scaled, y_train)

# --- 5. 模型評估 ---
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R-squared (R2): {r2:.4f} (越接近 1 越好)")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")

# --- 6. 解析權重 (Feature Importance) ---
# 因為做了標準化，這裡的係數大小直接代表重要性
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\n特徵重要性排行 (絕對值越大影響越大):")
print(feature_importance)

# 視覺化權重
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance (Standardized Coefficients)')
plt.show()

# --- 7. 結果視覺化 (三大圖表) ---

# (A) 預測值 vs 真實值 (必備)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted')
plt.show()

# (B) 殘差圖 (Day 02/05 重點：檢查是否有規律)
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot (Should be random)')
plt.show()

# (C) 地理空間圖 (僅限含有經緯度的資料集)
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, x="Longitude", y="Latitude", 
        size=df['Population']/100 if 'Population' in df.columns else None,
        hue="Target", palette="viridis", alpha=0.5
    )
    plt.title("Geospatial Analysis")
    plt.show()
else:
    print("此資料集無經緯度資訊，跳過地圖繪製。")