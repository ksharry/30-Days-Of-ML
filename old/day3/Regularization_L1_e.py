import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. 資料準備 (保持與您之前一致的環境) ---
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MedHouseVal')

# 取前 200 筆製造過擬合環境
subset_size = 200
X_subset = X.iloc[:subset_size]
y_subset = y.iloc[:subset_size]

X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42
)

# 多項式特徵擴充 (Degree=3) -> 標準化
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# --- 2. 自動搜尋最佳 Alpha (Grid Search) ---
# 設定搜尋範圍 (根據您圖表的 X 軸範圍)
alphas = np.logspace(-4, 0, 100) 

best_alpha = None
best_test_r2 = -np.inf
best_model = None

# 跑迴圈尋找 Test R2 最高的點
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=20000, tol=0.01)
    lasso.fit(X_train_scaled, y_train)
    
    # 這裡我們直接看測試集分數來選最好的 (呼應您的圖表邏輯)
    current_test_r2 = lasso.score(X_test_scaled, y_test)
    
    if current_test_r2 > best_test_r2:
        best_test_r2 = current_test_r2
        best_alpha = alpha
        best_model = lasso

# --- 3. 計算並列印最佳模型的詳細成績 ---
# 使用找到的最佳模型進行預測
y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled)

# 計算 Metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 計算留下的特徵數
n_features = np.sum(best_model.coef_ != 0)

print(f"=== 最佳 L1 模型搜尋結果 ===")
print(f"最佳 Alpha (參數): {best_alpha:.5f}")
print(f"保留特徵數: {n_features} / {X_train_scaled.shape[1]}")
print("-" * 30)
print(f"【訓練集 Train】(給模型看過的)")
print(f"  R2 Score (越接近 1 越好): {train_r2:.4f}")
print(f"  MSE (越接近 0 越好):      {train_mse:.4f}")
print("-" * 30)
print(f"【測試集 Test】(考試成績 - 您圖上的高點)")
print(f"  R2 Score (越接近 1 越好): {test_r2:.4f}")
print(f"  MSE (越接近 0 越好):      {test_mse:.4f}")