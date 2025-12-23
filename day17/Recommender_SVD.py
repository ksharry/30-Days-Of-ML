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
# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'ml-100k')
ZIP_FILE = os.path.join(SCRIPT_DIR, 'ml-100k.zip')
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 下載 MovieLens 100k 資料集
url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
if not os.path.exists(DATA_DIR):
    print(f"Downloading MovieLens data from {url}...")
    try:
        import urllib.request
        import zipfile
        urllib.request.urlretrieve(url, ZIP_FILE)
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(SCRIPT_DIR)
        print("Download and extraction complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        exit()

# 讀取資料
# u.data: User ID, Item ID, Rating, Timestamp
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(os.path.join(DATA_DIR, 'u.data'), sep='\t', names=ratings_cols, encoding='latin-1')

# u.item: Movie ID, Movie Title, ... (其他欄位忽略)
movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv(os.path.join(DATA_DIR, 'u.item'), sep='|', names=movies_cols, usecols=range(5), encoding='latin-1')

# 合併資料 (只取需要的欄位)
df = pd.merge(ratings, movies, on='movie_id')
print("MovieLens Data Loaded:")
print(df[['user_id', 'title', 'rating']].head())
print(f"Total Ratings: {len(df)}")
print(f"Total Users: {df.user_id.nunique()}")
print(f"Total Movies: {df.movie_id.nunique()}")

# 轉成矩陣形式 (User-Item Matrix)
# 這裡會產生一個 943 x 1682 的大矩陣，很多格子是 0 (稀疏)
R_df = df.pivot_table(index='user_id', columns='title', values='rating').fillna(0)
print("\nUser-Item Matrix Shape:", R_df.shape)

# 轉成 numpy array 並去中心化
R = R_df.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# --- 2. 矩陣分解 (Matrix Factorization via SVD) ---
# 對於真實數據，k (潛在特徵數) 通常設大一點，例如 50
k = 50
U, sigma, Vt = svds(R_demeaned, k=k)
sigma = np.diag(sigma)

print(f"\nSVD Done. k={k}")

# --- 3. 重建矩陣與預測 (Reconstruction & Prediction) ---
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns, index=R_df.index)

# --- 4. 推薦電影 (Recommendation) ---
def recommend_movies(user_id, num_recommendations=5):
    # 1. 取得該使用者的預測評分，並排序
    # user_id 是 1-based，index 是 0-based，所以要減 1
    user_row_number = user_id - 1 
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)
    
    # 2. 取得該使用者「已經看過」的電影
    user_data = ratings[ratings.user_id == user_id]
    user_full = (user_data.merge(movies, how = 'left', left_on = 'movie_id', right_on = 'movie_id').
                     sort_values(['rating'], ascending=False)
                 )
    
    print(f"\nUser {user_id} has already rated {len(user_full)} movies.")
    print("Top 3 favorite movies (Rated 5):")
    print(user_full[['title', 'rating']].head(3))
    
    # 3. 推薦「沒看過」且「預測分最高」的電影
    # 先把預測結果轉成 DataFrame 並清楚命名
    preds_df_user = pd.DataFrame(sorted_user_predictions).reset_index()
    preds_df_user.columns = ['title', 'Predictions'] # 強制命名，避免 index 混淆

    recommendations = (movies[~movies['movie_id'].isin(user_full['movie_id'])].
         merge(preds_df_user, how = 'left',
               left_on = 'title',
               right_on = 'title').
         sort_values('Predictions', ascending = False).
         iloc[:num_recommendations]
        )

    return recommendations

# 推薦給 User 1
user_id_to_recommend = 1
recs = recommend_movies(user_id_to_recommend)
print(f"\nTop 5 Recommendations for User {user_id_to_recommend}:")
print(recs[['title', 'Predictions']])

# --- 5. 視覺化潛在特徵 (Visualizing Latent Features) ---
# 因為電影太多了 (1682部)，我們只畫「評分次數最多」的前 20 部電影
# 這樣圖才不會密密麻麻
movie_stats = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})
top_movies_ids = movie_stats['rating']['size'].nlargest(20).index
top_movies_titles = movies[movies['movie_id'].isin(top_movies_ids)]['title'].values

# 我們只取前兩個維度來畫圖 (雖然模型用了 50 個)
# 注意：這只是為了視覺化，實際上高維度的關係更複雜
V_subset = Vt.T[:, :2] # 取前兩個特徵
# 找出這 20 部電影在矩陣中的位置
top_movies_indices = [R_df.columns.get_loc(t) for t in top_movies_titles]

plt.figure(figsize=(12, 10))
plt.scatter(V_subset[top_movies_indices, 0], V_subset[top_movies_indices, 1], alpha=0.5)

for i in top_movies_indices:
    x = V_subset[i, 0]
    y = V_subset[i, 1]
    title = R_df.columns[i]
    plt.text(x, y, title, fontsize=10)

plt.title('Latent Space Visualization (Top 20 Movies)')
plt.xlabel('Latent Feature 1')
plt.ylabel('Latent Feature 2')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '17-1_Latent_Space.png'))
print("Latent space plot saved.")

# 印出座標以供觀察
print("\nTop 20 Movies Coordinates (2D):")
for i in top_movies_indices:
    print(f"{R_df.columns[i]}: ({V_subset[i, 0]:.4f}, {V_subset[i, 1]:.4f})")

# --- 6. 模型驗證 (Evaluation: RMSE) ---
# 為了知道模型準不準，我們計算 RMSE (均方根誤差)
# 也就是比較「真實評分」和「預測評分」的差距
# 注意：嚴謹的做法應該要切分 Train/Test，這裡我們計算的是「重建誤差 (Reconstruction Error)」

# 只計算原本有評分的地方 (不計算原本是 0 的格子)
prediction_flattened = preds_df.values.flatten()
original_flattened = R_df.values.flatten()
nonzero_index = original_flattened.nonzero() # 找出非 0 的位置

original_nonzero = original_flattened[nonzero_index]
prediction_nonzero = prediction_flattened[nonzero_index]

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(original_nonzero, prediction_nonzero))

print(f"\nModel Evaluation (RMSE): {rmse:.4f}")
print("這代表我們的預測評分平均誤差約為 {:.2f} 分。".format(rmse))

# 舉個例子驗證
u_id, m_title = 1, 'Cinema Paradiso (1988)' # User 1 給這部片 5 分
real_rating = R_df.loc[u_id, m_title]
pred_rating = preds_df.loc[u_id, m_title]
print(f"\nExample Check for User {u_id}, Movie '{m_title}':")
print(f"Real Rating: {real_rating}")
print(f"Predicted:   {pred_rating:.2f}")
print(f"Diff:        {abs(real_rating - pred_rating):.2f}")
