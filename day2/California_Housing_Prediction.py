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
plt.show()

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

# --- 6. 解析權重 ---
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)
print(feature_importance)

# --- 7. 結果視覺化(True vs Predicted House Values) ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # 完美預測線
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted House Values')
plt.show()

# --- 7. 結果視覺化 (觀察天花板效應) ---
plt.figure(figsize=(10, 6))
sns.histplot(df['Target'], bins=50, kde=True)
plt.axvline(x=5.0, color='red', linestyle='--', label='Ceiling ($500k)')
plt.title('Distribution of House Prices')
plt.legend()
plt.show()

# --- 7. 結果視覺化 地理空間視覺化 (Geospatial Visualization) ---
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df, 
    x="Longitude", 
    y="Latitude", 
    size="Population",    # 點的大小代表人口數
    hue="Target",         # 點的顏色代表房價
    palette="viridis",    # 顏色風格
    alpha=0.5,            # 透明度
    sizes=(10, 200),      # 點的大小範圍
)
plt.title("California Housing Prices: Location & Population", fontsize=15)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="House Value", loc="upper right")
plt.show()

# --- 7. 結果視覺化 殘差分析 (Residual Analysis) ---
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--', lw=2)  # 畫出 0 的基準線
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.show()