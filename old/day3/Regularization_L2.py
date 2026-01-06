import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# --- 1. 資料準備 (維持過擬合環境) ---
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MedHouseVal')

# 取前 200 筆
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

# --- 2. L2 (Ridge) 迴圈調參 ---
# Ridge 的 Alpha 通常需要比 Lasso 大很多才會有感，所以範圍設大一點 (0.01 到 10000)
alphas = np.logspace(-2, 4, 50) 

train_scores = []
test_scores = []
avg_coef_magnitude = [] # 記錄係數的平均絕對值大小 (看 L2 壓制力)

for a in alphas:
    # Ridge 不會歸零，所以不用看 features count，改看係數大小
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_scaled, y_train)
    
    train_scores.append(ridge.score(X_train_scaled, y_train))
    test_scores.append(ridge.score(X_test_scaled, y_test))
    
    # 計算平均係數大小 (用絕對值平均)
    avg_coef = np.mean(np.abs(ridge.coef_))
    avg_coef_magnitude.append(avg_coef)

# --- 3. 視覺化 ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左軸：繪製 R2 分數
ax1.set_xlabel('Alpha (Regularization Strength) - Log Scale')
ax1.set_ylabel('R2 Score', color='tab:blue')
ax1.semilogx(alphas, train_scores, label='Train R2', color='tab:blue', linestyle='--')
ax1.semilogx(alphas, test_scores, label='Test R2', color='tab:blue', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim(-0.5, 1.1)

# 右軸：繪製平均係數大小 (觀察 Shrinkage)
ax2 = ax1.twinx()  
ax2.set_ylabel('Avg Coefficient Magnitude', color='tab:red')
ax2.semilogx(alphas, avg_coef_magnitude, label='Avg Coef Size', color='tab:red', linewidth=2, linestyle='-.')
ax2.tick_params(axis='y', labelcolor='tab:red')

# 標示最佳點
best_idx = np.argmax(test_scores)
best_alpha = alphas[best_idx]
best_score = test_scores[best_idx]
best_coef_size = avg_coef_magnitude[best_idx]

plt.title(f'Ridge (L2) Tuning: Best Alpha={best_alpha:.2f} (R2={best_score:.2f})')
fig.tight_layout()
plt.show()