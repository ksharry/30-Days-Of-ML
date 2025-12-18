# Day 04: 正則化回歸 (Regularization) - 加州房價預測
# ---------------------------------------------------------
# 這一天的目標是解決過擬合 (Overfitting) 問題。
# 我們將比較 Linear Regression, Ridge (L2), Lasso (L1) 的表現。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 模型與評估工具
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. 載入資料 (Data Loading) ---
# 使用 Scikit-Learn 內建的加州房價資料集
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

print(f"資料集維度: {df.shape}")
print(df.head())

# --- 2. 數據觀察 (EDA) ---
# 觀察目標變數的分佈
plt.figure(figsize=(8, 6))
sns.histplot(df['Target'], bins=30, kde=True)
plt.title('Distribution of House Prices')
pic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pic')
os.makedirs(pic_dir, exist_ok=True)
plt.savefig(os.path.join(pic_dir, '4-1_Distribution.png'))
# plt.show()

# --- 3. 資料分割與前處理 ---
X = df.drop('Target', axis=1)
y = df['Target']

# 切分資料 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化 (Standardization) - 對於正則化模型非常重要！
# 因為正則化是懲罰係數的大小，如果特徵尺度不同，係數大小就沒有可比性
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. 建立與訓練模型 (比較三種) ---
models = {
    'Linear': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=0.01) # alpha 不能設太大，否則所有係數都會變 0
}

results = {}

print("\n模型評估結果:")
print(f"{'Model':<12} | {'MSE':<10} | {'R2':<8}")
print("-" * 35)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'model': model, 'mse': mse, 'r2': r2, 'pred': y_pred}
    print(f"{name:<12} | {mse:.4f}     | {r2:.4f}")

# --- 5. 解析權重 (Feature Importance Comparison) ---
# 比較 Linear, Ridge, Lasso 的係數
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Linear': results['Linear']['model'].coef_,
    'Ridge': results['Ridge']['model'].coef_,
    'Lasso': results['Lasso']['model'].coef_
})

print("\n係數比較 (注意 Lasso 是否將某些係數變為 0):")
print(coef_df)

# 視覺化係數差異
plt.figure(figsize=(12, 6))
coef_melted = coef_df.melt(id_vars='Feature', var_name='Model', value_name='Coefficient')
sns.barplot(x='Feature', y='Coefficient', hue='Model', data=coef_melted)
plt.title('Coefficient Comparison: Linear vs Ridge vs Lasso')
plt.xticks(rotation=45)
plt.savefig(os.path.join(pic_dir, '4-2_Coefficients.png'))
# plt.show()

# --- 6. 結果視覺化 (殘差圖比較) ---
plt.figure(figsize=(15, 5))

for i, (name, res) in enumerate(results.items()):
    plt.subplot(1, 3, i+1)
    y_pred = res['pred']
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.title(f'{name} Residuals')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')

plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '4-3_Residuals_Comparison.png'))
# plt.show()
