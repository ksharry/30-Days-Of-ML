import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# --- 1. 資料準備 (標準化過擬合環境) ---
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MedHouseVal')

subset_size = 200
X_subset = X.iloc[:subset_size]
y_subset = y.iloc[:subset_size]

X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42
)

# 多項式特徵 (Degree=3) -> 標準化
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# --- 2. 雙參數網格搜尋 (Grid Search) ---
# 設定 Alpha 範圍 (縱軸)
alphas = np.logspace(-4, 0, 20) 
# 設定 L1_ratio 範圍 (橫軸) - 從接近 Ridge (0.1) 到接近 Lasso (0.99)
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

results = []

print("正在掃描參數組合 (這可能需要幾秒鐘)...")

for a in alphas:
    row = []
    for r in l1_ratios:
        # 建立 ElasticNet 模型
        el = ElasticNet(alpha=a, l1_ratio=r, max_iter=10000, tol=0.01)
        el.fit(X_train_scaled, y_train)
        score = el.score(X_test_scaled, y_test)
        row.append(score)
    results.append(row)

# 轉成 DataFrame 以便繪圖
results_df = pd.DataFrame(results, index=np.round(alphas, 5), columns=l1_ratios)

# --- 3. 繪製熱力圖 (Heatmap) ---
plt.figure(figsize=(10, 8))
sns.heatmap(results_df, annot=False, cmap='viridis', cbar_kws={'label': 'Test R2 Score'})

# 將 Y 軸 (Alpha) 反轉，讓小的在下面，大的在上面，比較符合直覺
plt.gca().invert_yaxis()

plt.title('ElasticNet Performance: Alpha vs L1_ratio\n(Brighter Color = Better Score)')
plt.xlabel('L1 Ratio (0=Ridge, 1=Lasso)')
plt.ylabel('Alpha (Regularization Strength)')
plt.tight_layout()
plt.show()