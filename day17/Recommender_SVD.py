# Day 17: 推薦系統 (Recommender System) - 電影推薦
# ---------------------------------------------------------
# 這一天的目標是實作一個簡單的「協同過濾 (Collaborative Filtering)」推薦系統。
# 我們會使用 SVD (奇異值分解) 來預測使用者可能會給某部電影打幾分。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.sparse.linalg import svds

# --- 1. 準備資料 (Data Preparation) ---
# 為了演示方便，我們手動建立一個小型的電影評分矩陣
# 真實世界通常使用 MovieLens 資料集
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 評分數據：User 對 Movie 的評分 (1-5分，0代表沒看過)
# 假設有 6 個使用者，5 部電影
ratings_dict = {
    'User': ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Charlie', 'Charlie', 'David', 'David', 'Eve', 'Frank'],
    'Movie': ['Matrix', 'Titanic', 'Avengers', 'Matrix', 'Avengers', 'Titanic', 'Frozen', 'Matrix', 'Frozen', 'Avengers', 'Titanic'],
    'Rating': [5, 3, 4, 5, 5, 2, 5, 4, 1, 3, 2]
}

df = pd.DataFrame(ratings_dict)
print("Raw Ratings Data:")
print(df.head())

# 轉成矩陣形式 (User-Item Matrix)
# index=User, columns=Movie, values=Rating
R_df = df.pivot(index='User', columns='Movie', values='Rating').fillna(0)
print("\nUser-Item Matrix (R):")
print(R_df)

# 轉成 numpy array
R = R_df.values
user_ratings_mean = np.mean(R, axis=1) # 每個使用者的平均評分
R_demeaned = R - user_ratings_mean.reshape(-1, 1) # 去中心化 (減去平均)

# --- 2. 矩陣分解 (Matrix Factorization via SVD) ---
# 我們想把這個稀疏矩陣 R 分解成 U, Sigma, Vt
# k=2 代表我們只取前 2 個潛在特徵 (Latent Features)
# 例如：這 2 個特徵可能代表「動作片程度」和「愛情片程度」
U, sigma, Vt = svds(R_demeaned, k=2)

sigma = np.diag(sigma) # 轉成對角矩陣

print(f"\nShape of U: {U.shape}")     # (Users, k)
print(f"Shape of Sigma: {sigma.shape}") # (k, k)
print(f"Shape of Vt: {Vt.shape}")    # (k, Movies)

# --- 3. 重建矩陣與預測 (Reconstruction & Prediction) ---
# 預測評分 = U * Sigma * Vt + 平均分
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns, index=R_df.index)
print("\nPredicted Ratings Matrix:")
print(preds_df.round(2))

# --- 4. 推薦電影 (Recommendation) ---
# 讓我們看看要推薦什麼給 'David'
user_name = 'David'
user_row_number = R_df.index.get_loc(user_name)
sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)

# 找出他已經看過的電影
user_data = df[df.User == user_name]
print(f"\nUser {user_name} has already rated:")
print(user_data)

print(f"\nRecommendations for {user_name}:")
# 推薦他沒看過，且預測分數最高的電影
recommendations = sorted_user_predictions.drop(user_data.Movie.tolist())
print(recommendations.head(3))

# --- 5. 視覺化潛在特徵 (Visualizing Latent Features) ---
# 我們把電影和使用者畫在同一個 2D 平面上
# 看看誰跟誰比較近
plt.figure(figsize=(10, 8))

# 畫電影 (基於 Vt 的轉置，即 V)
# Vt 是 (k, Movies)，所以 V 是 (Movies, k)
V = Vt.T
plt.scatter(V[:, 0], V[:, 1], c='red', marker='x', s=100, label='Movies')
for i, txt in enumerate(R_df.columns):
    plt.text(V[i, 0]+0.02, V[i, 1], txt, fontsize=12, color='red')

# 畫使用者 (基於 U)
plt.scatter(U[:, 0], U[:, 1], c='blue', marker='o', s=100, label='Users')
for i, txt in enumerate(R_df.index):
    plt.text(U[i, 0]+0.02, U[i, 1], txt, fontsize=12, color='blue')

plt.title('Latent Space Visualization (SVD)')
plt.xlabel('Latent Feature 1')
plt.ylabel('Latent Feature 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '17-1_Latent_Space.png'))
print("Latent space plot saved.")
