import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# --- 1. 準備數據 (與您原本一致，製造過擬合環境) ---
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MedHouseVal')

# 只取前 200 筆以製造 Overfitting
subset_size = 200
X_subset = X.iloc[:subset_size]
y_subset = y.iloc[:subset_size]

X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42
)

# --- 2. 建立多項式特徵 (擴充特徵數量) ---
# degree=3 會讓 8 個特徵變成 164 個特徵 (爆炸性增長)
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 取得所有特徵的名稱 (為了知道留下來的是誰)
feature_names = poly.get_feature_names_out(X.columns)
print(f"擴充後的特徵總數: {len(feature_names)}")

# 標準化 (Lasso 對尺度非常敏感，必須做！)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# --- 3. 第一部分：查看特定 Alpha 下的「倖存者」 ---
specific_alpha = 0.01
lasso = Lasso(alpha=specific_alpha, max_iter=20000, tol=0.01) # tol放寬加速收斂
lasso.fit(X_train_scaled, y_train)

# 找出係數不為 0 的特徵
coefs = lasso.coef_
nonzero_mask = coefs != 0
survivors = feature_names[nonzero_mask]
survivor_coefs = coefs[nonzero_mask]

print(f"\n=== 當 Alpha = {specific_alpha} 時的結果 ===")
print(f"原本有 {len(feature_names)} 個特徵 -> Lasso 刪除後剩下 {len(survivors)} 個特徵")
print(f"訓練集 R2: {lasso.score(X_train_scaled, y_train):.4f}")
print(f"測試集 R2: {lasso.score(X_test_scaled, y_test):.4f}")

print("\n--- 倖存且權重最大的前 5 個特徵 (正負影響) ---")
# 建立 DataFrame 方便查看
df_coeffs = pd.DataFrame({'Feature': survivors, 'Weight': survivor_coefs})
# 依權重絕對值排序
df_coeffs['Abs_Weight'] = df_coeffs['Weight'].abs()
print(df_coeffs.sort_values(by='Abs_Weight', ascending=False).head(5))


# --- 4. 第二部分：Alpha 調參路徑 (如何影響結果) ---
# 設定一組不同的 alpha 值，從極小(像 Linear) 到極大(全部砍光)
alphas = np.logspace(-4, 0, 50) # 產生 0.0001 到 1.0 的 50 個數字
train_scores = []
test_scores = []
n_features_kept = []

for a in alphas:
    l = Lasso(alpha=a, max_iter=10000, tol=0.01)
    l.fit(X_train_scaled, y_train)
    
    train_scores.append(l.score(X_train_scaled, y_train))
    test_scores.append(l.score(X_test_scaled, y_test))
    n_features_kept.append(np.sum(l.coef_ != 0))

# --- 5. 視覺化調參過程 ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# 繪製 R2 分數 (左軸)
ax1.set_xlabel('Alpha (Regularization Strength) - Log Scale')
ax1.set_ylabel('R2 Score', color='tab:blue')
ax1.semilogx(alphas, train_scores, label='Train R2', color='tab:blue', linestyle='--')
ax1.semilogx(alphas, test_scores, label='Test R2', color='tab:blue', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim(-0.5, 1.1)

# 繪製剩餘特徵數 (右軸)
ax2 = ax1.twinx()  
ax2.set_ylabel('Number of Features Kept', color='tab:red')
ax2.semilogx(alphas, n_features_kept, label='Features Kept', color='tab:red', linewidth=2, linestyle='-.')
ax2.tick_params(axis='y', labelcolor='tab:red')

# 標示最佳點
best_idx = np.argmax(test_scores)
best_alpha = alphas[best_idx]
best_score = test_scores[best_idx]
best_feat_count = n_features_kept[best_idx]

plt.title(f'Lasso Tuning: Best Alpha={best_alpha:.4f} (R2={best_score:.2f}, Feats={best_feat_count})')
fig.tight_layout()
plt.show()