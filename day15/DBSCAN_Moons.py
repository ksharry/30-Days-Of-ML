# Day 15: DBSCAN 密度聚類 (Density-Based Spatial Clustering of Applications with Noise)
# ---------------------------------------------------------
# 這一天的目標是解決 K-Means 的死穴：不規則形狀與噪聲。
# 我們使用 "月亮形狀" (Moons) 數據集來展示 DBSCAN 的強大之處。
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

# --- 1. 準備資料 (Data Preparation) ---
# 產生兩個半月形的數據，並加入一些噪聲
# 這是 K-Means 最不擅長的形狀 (因為它假設群聚是圓形的)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

# --- 2. K-Means 的失敗案例 (K-Means Failure) ---
#我們先用 K-Means 跑一次，看看它會怎麼分
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
plt.title('K-Means Clustering (Failed)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
# 畫出 K-Means 的決策邊界 (直線)
# 這裡簡單畫個示意，K-Means 傾向於畫直線切分

# --- 3. DBSCAN 的成功案例 (DBSCAN Success) ---
# DBSCAN 需要兩個參數：
# eps (epsilon): 半徑，鄰居要在多近才算「同一群」。
# min_samples: 最小樣本數，半徑內至少要有幾個點才算「核心點」。

# 參數調整是 DBSCAN 的難點，這裡我們設 eps=0.3, min_samples=5
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

# 視覺化
plt.subplot(1, 2, 2)
# DBSCAN 的標籤中，-1 代表噪聲 (Noise/Outliers)，我們用紅色標示
unique_labels = set(y_dbscan)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1] # 黑色代表噪聲
        label = 'Noise'
    else:
        label = f'Cluster {k}'

    class_member_mask = (y_dbscan == k)
    
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=10 if k != -1 else 6, label=label)

plt.title('DBSCAN Clustering (Success)')
plt.xlabel('Feature 1')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '15-1_KMeans_vs_DBSCAN.png'))
print("Comparison plot saved.")

# --- 4. 參數敏感度測試 (Parameter Sensitivity) ---
# 讓我們看看不同的 eps 對結果的影響
eps_values = [0.1, 0.3, 0.5]
plt.figure(figsize=(15, 4))

for i, eps in enumerate(eps_values):
    db = DBSCAN(eps=eps, min_samples=5)
    y_db = db.fit_predict(X)
    
    plt.subplot(1, 3, i+1)
    plt.scatter(X[:, 0], X[:, 1], c=y_db, cmap='plasma', s=50)
    plt.title(f'DBSCAN (eps={eps})')
    
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '15-2_DBSCAN_Eps_Sensitivity.png'))
print("Sensitivity plot saved.")
