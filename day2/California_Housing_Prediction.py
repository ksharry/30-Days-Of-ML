import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. 載入資料 ---
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Target'] = housing.target  # 加入目標變數 (房價)

print("資料集維度:", df.shape)
print(df.head())

# --- 2. 數據觀察 (EDA) ---
# 觀察「收入中位數 (MedInc)」與「房價 (Target)」的關係
plt.figure(figsize=(8, 6))
sns.scatterplot(x='MedInc', y='Target', data=df, alpha=0.3)
plt.title('Median Income vs House Value')
plt.xlabel('Median Income')
plt.ylabel('House Value')
plt.show()

# 房價分佈圖 (觀察天花板效應)
plt.figure(figsize=(10, 6))

# 繪製直方圖搭配密度曲線 (KDE)
# bins=50 表示將資料切成 50 個區間，切越細看得越清楚
sns.histplot(df['Target'], bins=50, kde=True, color='#4c72b0')

# 加上一條紅色的垂直虛線在 5.0 的位置
plt.axvline(x=5.0, color='red', linestyle='--', linewidth=2, label='Ceiling ($500k)')

# 加入標題與標籤
plt.title('Distribution of California House Prices', fontsize=14)
plt.xlabel('Median House Value (in $100,000 units)', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)
plt.legend() # 顯示圖例

plt.grid(axis='y', alpha=0.3) # 加入淡淡的水平網格線增加易讀性
plt.show()

# ==========================================
# --- 3. 資料分割 ---
# 將資料切分為 80% 訓練集，20% 測試集
X = df.drop('Target', axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. 建立與訓練模型 ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- 5. 模型評估 ---
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2 Score): {r2:.4f}")

# --- 6. 解析權重 (Weights Interpretation) ---
# 查看哪些特徵對房價影響最大
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\n特徵權重 (係數):")
print(feature_importance)

# --- 7. 結果視覺化 ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # 完美預測線
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted House Values')
plt.show()