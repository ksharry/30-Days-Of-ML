# Day 13: K-Means 聚類 (K-Means Clustering) - 商場客戶分群
# ---------------------------------------------------------
# 這一天的目標是進入「非監督式學習」的世界。
# 我們沒有標準答案 (Labels)，而是要讓演算法自己發現資料中的結構。
# 案例：根據「年收入」與「消費分數」將商場客戶分群。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'Mall_Customers.csv')
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 檢查資料是否存在
if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found.")
    print("Please download 'Mall_Customers.csv' from Kaggle and place it in the day13 folder.")
    exit()

# 讀取資料
df = pd.read_csv(DATA_FILE)
print("Data loaded:")
print(df.head())

# 我們只取兩個特徵來做 2D 視覺化：年收入 vs 消費分數
# index 3: Annual Income (k$), index 4: Spending Score (1-100)
X = df.iloc[:, [3, 4]].values

# --- 2. 決定 K 值：手肘法 (Elbow Method) ---
# K-Means 需要我們先告訴它要分幾群 (K)。
# 我們計算不同 K 值下的 WCSS (Within-Cluster Sum of Square，群內誤差平方和)。
# WCSS 越小代表分得越緊密，但 K 越大 WCSS 本來就會越小。
# 我們要找的是「轉折點」(Elbow)，即 WCSS 下降幅度開始變緩的地方。

wcss = []
k_range = range(1, 11)

for i in k_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # inertia_ 就是 WCSS

plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.xticks(k_range)
plt.grid(True)
plt.savefig(os.path.join(pic_dir, '13-1_Elbow_Method.png'))
print("Elbow Method plot saved.")

# --- 3. 訓練模型 (Model Training) ---
# 根據手肘法 (通常 Mall Dataset 的轉折點在 K=5)
optimal_k = 5
print(f"Training K-Means with K={optimal_k}...")

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# --- 4. 視覺化結果 (Visualization) ---
plt.figure(figsize=(10, 7))

# 畫出每一群的點
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']
# 這裡假設 K=5，如果不是 5 可能會報錯，但為了教學演示我們先固定
# 為了通用性，我們用迴圈畫
for i in range(optimal_k):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], 
                s=100, c=colors[i] if i < len(colors) else None, 
                label=f'Cluster {i+1}')

# 畫出質心 (Centroids)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='yellow', label='Centroids', edgecolors='black')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(pic_dir, '13-2_Clusters.png'))
print("Cluster plot saved.")

# --- 5. 輸出結果解釋 ---
# 簡單分析每一群的特性
print("\n--- Cluster Analysis ---")
for i in range(optimal_k):
    cluster_data = X[y_kmeans == i]
    mean_income = cluster_data[:, 0].mean()
    mean_score = cluster_data[:, 1].mean()
    print(f"Cluster {i+1}: Avg Income = {mean_income:.1f}k$, Avg Score = {mean_score:.1f}")
