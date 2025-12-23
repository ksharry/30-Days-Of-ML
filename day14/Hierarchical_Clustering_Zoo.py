# Day 14: 層次聚類 (Hierarchical Clustering) - 脊椎動物分類
# ---------------------------------------------------------
# 這一天的目標是學習另一種強大的聚類方法：層次聚類。
# 不同於 K-Means 需要預先指定 K，層次聚類會建立一個「樹狀結構」，
# 讓我們可以從不同層次去觀察資料的關聯性。
# 案例：根據動物的特徵 (如體溫、胎生、水生等) 畫出分類樹 (Dendrogram)。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'vertebrate.csv')
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 我們手動建立一個小型的脊椎動物資料集
# 這樣畫出來的樹狀圖才看得清楚每個動物的名字
data = {
    'Name': ['Human', 'Python', 'Salmon', 'Whale', 'Frog', 'Komodo', 'Bat', 'Pigeon', 'Cat', 'Leopard Shark', 'Turtle', 'Penguin', 'Porcupine', 'Eel', 'Salamander'],
    'Warm-blooded': [1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0], # 恆溫
    'Gives Birth':  [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0], # 胎生
    'Aquatic':      [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1], # 水生
    'Aerial':       [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # 飛翔
    'Has Legs':     [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], # 有腳
    'Hibernates':   [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]  # 冬眠
}

df = pd.DataFrame(data)
df.to_csv(DATA_FILE, index=False)
print("Vertebrate data created:")
print(df)

# 取出特徵 (不包含名字)
X = df.drop('Name', axis=1)

# --- 2. 繪製樹狀圖 (Dendrogram) ---
# 這是層次聚類最精華的部分！
# linkage 函數會計算兩兩樣本之間的距離，並把它們「黏」在一起
# method='ward' 是一種最小化變異數的黏合策略 (類似 K-Means 的精神)
linked = linkage(X, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=df['Name'].values, # 把動物名字標上去
            distance_sort='descending',
            show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Vertebrates)')
plt.xlabel('Species')
plt.ylabel('Euclidean Distance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '14-1_Dendrogram.png'))
print("Dendrogram saved.")

# --- 3. 訓練模型 (Agglomerative Clustering) ---
# 透過觀察樹狀圖，我們可以決定要切幾刀 (幾群)
# 假設我們想分成 3 大類 (哺乳類/鳥類, 爬蟲/兩棲, 魚類?)
n_clusters = 3
cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
y_pred = cluster.fit_predict(X)

# 將分群結果加回 DataFrame
df['Cluster'] = y_pred

print(f"\n--- Clustering Results (K={n_clusters}) ---")
for i in range(n_clusters):
    print(f"\nCluster {i}:")
    print(df[df['Cluster'] == i]['Name'].values)

# --- 4. 視覺化分群結果 (Heatmap) ---
# 因為特徵是 0/1 的類別型數據，用 Heatmap 來看每一群的特徵分佈最適合
# 我們把資料依據 Cluster 排序
df_sorted = df.sort_values('Cluster')
X_sorted = df_sorted.drop(['Name', 'Cluster'], axis=1)

plt.figure(figsize=(8, 6))
sns.heatmap(X_sorted, annot=True, cmap='coolwarm', cbar=False, yticklabels=df_sorted['Name'])
plt.title('Features Heatmap sorted by Cluster')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '14-2_Cluster_Heatmap.png'))
print("Heatmap saved.")
