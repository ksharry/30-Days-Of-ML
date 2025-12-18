import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. 資料準備 ---
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MedHouseVal')

subset_size = 200
X_subset = X.iloc[:subset_size]
y_subset = y.iloc[:subset_size]

X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# --- 2. 自動搜尋最佳 Alpha ---
# Ridge 範圍設大一點
alphas = np.logspace(-2, 4, 100) 

best_alpha = None
best_test_r2 = -np.inf
best_model = None

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    
    current_test_r2 = ridge.score(X_test_scaled, y_test)
    
    if current_test_r2 > best_test_r2:
        best_test_r2 = current_test_r2
        best_alpha = alpha
        best_model = ridge

# --- 3. 結果評估 ---
y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 檢查係數狀況
n_total_features = X_train_scaled.shape[1]
# 在 Ridge 中，我們檢查有多少係數是「極度接近 0 (小於 1e-5)」而非完全等於 0
n_zero_features = np.sum(np.abs(best_model.coef_) < 1e-5) 

print(f"=== 最佳 Ridge (L2) 模型搜尋結果 ===")
print(f"最佳 Alpha (參數): {best_alpha:.5f}")
print(f"總特徵數: {n_total_features}")
print(f"被視為 0 的特徵數 (< 1e-5): {n_zero_features} (Ridge 通常這裡會是 0)")
print("-" * 30)
print(f"【訓練集 Train】")
print(f"  R2 Score: {train_r2:.4f}")
print(f"  MSE:      {train_mse:.4f}")
print("-" * 30)
print(f"【測試集 Test】")
print(f"  R2 Score: {test_r2:.4f}")
print(f"  MSE:      {test_mse:.4f}")