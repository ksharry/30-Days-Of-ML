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

# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'zoo.csv')
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 下載 UCI Zoo Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"
if not os.path.exists(DATA_FILE):
    print(f"Downloading data from {url}...")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, DATA_FILE)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        exit()

# 定義欄位名稱 (UCI 資料集沒有 Header)
cols = ['animal_name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 
        'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 
        'fins', 'legs', 'tail', 'domestic', 'catsize', 'class_type']

df = pd.read_csv(DATA_FILE, names=cols)
print("Zoo data loaded:")
print(df.head())

# --- 2. 繪製樹狀圖 (Dendrogram) ---
# 為了讓圖表清晰，我們只隨機抽取 40 隻動物來畫樹狀圖
df_sample = df.sample(n=40, random_state=42)
X_sample = df_sample.drop(['animal_name', 'class_type'], axis=1)
names_sample = df_sample['animal_name'].values
print("Sampled animals:", names_sample)

linked = linkage(X_sample, method='ward')

plt.figure(figsize=(12, 7))
dendrogram(linked,
            orientation='top',
            labels=names_sample,
            distance_sort='descending',
            show_leaf_counts=True,
            leaf_rotation=90,
            leaf_font_size=10)

# 畫出「切一刀」的示意線 (假設我們在距離=10的地方切)
# 這條線穿過了幾條垂直線，就代表分成了幾群
threshold = 10
plt.axhline(y=threshold, color='r', linestyle='--')
plt.text(x=5, y=threshold + 0.5, s=f'Cut at Distance={threshold}', color='r', fontsize=12)

plt.title('Hierarchical Clustering Dendrogram (Red Line = The "Cut")')
plt.xlabel('Animal Name')
plt.ylabel('Euclidean Distance')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '14-1_Dendrogram.png'))
print("Dendrogram saved.")

# --- 3. 訓練模型 (Agglomerative Clustering) ---
# 使用完整資料集 (101 隻動物)
X = df.drop(['animal_name', 'class_type'], axis=1)

# 真實世界有 7 大類 (哺乳、鳥、爬蟲、魚、兩棲、昆蟲、無脊椎)
# 我們看看演算法能不能自己發現這 7 群
n_clusters = 7
cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
y_pred = cluster.fit_predict(X)

df['Cluster'] = y_pred

# --- 4. 評估結果 (與真實類別比較) ---
# 因為我們有真實答案 (class_type)，可以做個交叉比對
# class_type: 1=Mammal, 2=Bird, 3=Reptile, 4=Fish, 5=Amphibian, 6=Bug, 7=Invertebrate
class_names = {1:'Mammal', 2:'Bird', 3:'Reptile', 4:'Fish', 5:'Amphibian', 6:'Bug', 7:'Invertebrate'}
df['Class_Name'] = df['class_type'].map(class_names)

print(f"\n--- Clustering Results (K={n_clusters}) ---")
# 建立混淆矩陣 (Crosstab) 來看每一群抓到了什麼
ct = pd.crosstab(df['Cluster'], df['Class_Name'])
print(ct)

# --- 5. 視覺化分群結果 (Heatmap) ---
# 排序以便觀察
df_sorted = df.sort_values('Cluster')
# 建立一個標籤欄位：Cluster - Name (例如 "1 - Lion")
df_sorted['Label'] = df_sorted['Cluster'].astype(str) + ' - ' + df_sorted['animal_name']

# 只取特徵欄位畫圖
X_sorted = df_sorted.drop(['animal_name', 'class_type', 'Class_Name', 'Cluster', 'Label'], axis=1)

plt.figure(figsize=(12, 14)) # 拉長一點才看得到所有動物
sns.heatmap(X_sorted, cmap='coolwarm', cbar=False, yticklabels=df_sorted['Label'])
plt.title('Features Heatmap sorted by Cluster')
plt.yticks(fontsize=8) 
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '14-2_Cluster_Heatmap.png'))
print("Heatmap saved.")
