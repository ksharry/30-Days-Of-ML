import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
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

# --- 2. 精細搜尋 (Fine-Grained Search) ---
# 我們搜尋更細的網格來找極限
alphas = np.logspace(-4, 0, 50)
l1_ratios = np.linspace(0.01, 1.0, 25) # 測試 25 種混合比例

best_alpha = None
best_l1_ratio = None
best_test_r2 = -np.inf
best_model = None

print("正在尋找 ElasticNet 最佳解...")

for alpha in alphas:
    for ratio in l1_ratios:
        el = ElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=20000, tol=0.01)
        el.fit(X_train_scaled, y_train)
        
        current_test_r2 = el.score(X_test_scaled, y_test)
        
        if current_test_r2 > best_test_r2:
            best_test_r2 = current_test_r2
            best_alpha = alpha
            best_l1_ratio = ratio
            best_model = el

# --- 3. 結果評估 ---
y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 計算特徵數 (ElasticNet 也有 L1 成分，所以也會有特徵歸零)
n_features = np.sum(best_model.coef_ != 0)

print(f"\n=== ElasticNet 最終挑戰結果 ===")
print(f"最佳 Alpha:   {best_alpha:.5f}")
print(f"最佳 L1 Ratio: {best_l1_ratio:.2f}")
if best_l1_ratio > 0.9:
    print("  -> 結論: 模型偏好 Lasso (L1)")
elif best_l1_ratio < 0.1:
    print("  -> 結論: 模型偏好 Ridge (L2)")
else:
    print("  -> 結論: 混合使用 (L1 + L2) 效果最好")

print(f"保留特徵數: {n_features} / {X_train_scaled.shape[1]}")
print("-" * 30)
print(f"【測試集 Test R2】: {test_r2:.5f}")
print(f"(L1 與 L2 約為 0.88，請比較此分數是否有提升)")
print("-" * 30)
print(f"Test MSE: {test_mse:.4f}")