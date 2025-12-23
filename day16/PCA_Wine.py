# Day 16: 主成分分析 (PCA) - 葡萄酒分類
# ---------------------------------------------------------
# 這一天的目標是學習「降維 (Dimensionality Reduction)」。
# 當資料特徵太多 (例如 13 種化學成分) 時，我們很難畫圖觀察，也很容易發生「維度災難」。
# PCA 可以幫我們把 13 維壓縮成 2 維，讓我們一眼看穿資料的結構！
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 載入 Wine Dataset (葡萄酒資料集)
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

print("Data loaded:")
print(f"Features: {feature_names}")
print(f"Shape: {X.shape}") # (178, 13) -> 178 瓶酒，13 種成分

# --- 2. 資料標準化 (Standardization) ---
# PCA 對數據的尺度非常敏感！
# 如果某個特徵數值很大 (例如 Proline 含量幾百)，它會主導整個 PCA。
# 所以一定要先做標準化 (Mean=0, Std=1)。
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. 執行 PCA (降維到 2D) ---
# 我們想把 13 維壓縮成 2 維，方便畫圖
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nOriginal shape: {X.shape}")
print(f"Reduced shape: {X_pca.shape}") # (178, 2)

# 解釋變異量 (Explained Variance)
# 這告訴我們：這 2 個新特徵 (PC1, PC2) 保留了原始資料多少的資訊量？
explained_variance = pca.explained_variance_ratio_
print(f"\nExplained Variance Ratio: {explained_variance}")
print(f"Total Information Retained: {sum(explained_variance) * 100:.2f}%")

# --- 4. 視覺化結果 (Visualization) ---

# 1. 2D 投影圖 (PCA Projection)
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Wine Dataset (13D -> 2D)')
plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}%)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '16-1_PCA_Projection.png'))
print("Projection plot saved.")

# 2. 累積解釋變異量 (Cumulative Explained Variance)
# 看看如果我們保留更多維度，能保留多少資訊？
pca_full = PCA(n_components=13)
pca_full.fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 14), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('How many components do we need?')
plt.grid(True)
# 畫一條 90% 的參考線
plt.axhline(y=0.9, color='r', linestyle='-')
plt.text(0.5, 0.92, '90% Threshold', color='red', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '16-2_Explained_Variance.png'))
print("Variance plot saved.")

# 3. 特徵權重圖 (Loading Plot / Heatmap)
# 讓我們看看 PC1 和 PC2 到底是由哪些原始特徵組成的？
# components_ 屬性儲存了特徵向量 (Eigenvectors)
plt.figure(figsize=(12, 6))
sns.heatmap(pca.components_, cmap='coolwarm', annot=True, xticklabels=feature_names, yticklabels=['PC1', 'PC2'])
plt.title('PCA Components (Feature Weights)')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '16-3_PCA_Components.png'))
print("Components heatmap saved.")
