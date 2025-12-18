# 模版三：非監督分群 (Unsupervised Clustering)
# 適用日子：Day 13-17 (K-Means, DBSCAN, PCA)
# ==========================================
# Day XX: [填入演算法名稱，如 K-Means Clustering]
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests

# 模型與評估工具
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
    print(f"讀取自定義資料失敗 ({e})，改用 make_blobs 產生虛擬資料...")
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=0)
    df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])

print(f"資料集維度: {df.shape}")
print(df.head())

# --- 2. 數據觀察 (EDA) ---
plt.figure(figsize=(8, 6))
# 假設前兩欄是主要特徵
sns.scatterplot(x=df.columns[0], y=df.columns[1], data=df, alpha=0.6)
plt.title('Raw Data Distribution')
plt.show()

# --- 3. 資料前處理 ---
# 分群算法對距離敏感，建議標準化
# 注意：非監督學習通常沒有 Target (y)，或者我們假裝不知道 y
if 'Target' in df.columns:
    X = df.drop('Target', axis=1)
else:
    X = df

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. 建立與訓練模型 ---
# 這裡通常包含決定 K 值的步驟 (如手肘法)
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled) # 直接分群並預測

# 將分群結果加回 DataFrame
df['Cluster'] = y_kmeans

# --- 5. 模型評估 ---
# 輪廓係數 (Silhouette Score): 越接近 1 越好
try:
    score = silhouette_score(X_scaled, y_kmeans)
    print(f"Silhouette Score: {score:.4f}")
except Exception as e:
    print(f"無法計算輪廓係數: {e}")

# --- 6. 解析權重 (質心分析) ---
# 對於 K-Means，我們可以看質心 (Centroids) 的位置
centers = scaler.inverse_transform(kmeans.cluster_centers_) # 轉回原始尺度
print("Cluster Centers (Original Scale):\n", centers)

# --- 7. 結果視覺化 (Clustering Result) ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df.columns[0], y=df.columns[1], hue='Cluster', data=df, palette='viridis', s=50)
# 畫出質心
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, label='Centroids')
plt.title('K-Means Clustering Result')
plt.legend()
plt.show()