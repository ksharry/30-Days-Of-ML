import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# --- 1. 準備數據 (Data Preparation) ---
# 載入加州房價資料集
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MedHouseVal')

# 【關鍵步驟】為了演示 Overfitting，我們故意只取少量的數據
# 數據越少，模型越容易死記硬背 (Overfit)
subset_size = 200  # 只取前 200 筆資料
X_subset = X.iloc[:subset_size]
y_subset = y.iloc[:subset_size]

# 切分訓練集與測試集 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42
)

print(f"原始特徵數量: {X_train.shape[1]}")

# --- 2. 建立模型管線 (Model Pipeline) ---
# 我們將 "特徵擴充" -> "標準化" -> "回歸模型" 包裝成一個 Pipeline
# degree=3 代表會產生 x^2, x^3 以及 x1*x2 這種交互作用項
degree = 3

# 定義三種模型
models = {
    "Linear (No Reg)": make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression()
    ),
    "Ridge (L2)": make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=10) # alpha 越大，懲罰越重
    ),
    "Lasso (L1)": make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Lasso(alpha=0.01, max_iter=10000) # alpha 越大，越容易把係數砍成 0
    )
}

# --- 3. 訓練與評估 (Train & Evaluate) ---
results = {}
coefficients = {}

print("-" * 30)
for name, model in models.items():
    # 訓練模型
    model.fit(X_train, y_train)
    
    # 預測
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 計算 R2 分數
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results[name] = {"Train R2": train_r2, "Test R2": test_r2}
    
    # 儲存係數 (最後一個步驟是回歸模型，取出其 coef_)
    # Lasso 因為會讓係數歸零，所以這裡特別重要
    if name != "Linear (No Reg)": # Linear 的係數太大，畫圖會破壞比例，故只存 L1/L2
        coefficients[name] = model.steps[-1][1].coef_

    # 取得擴充後的特徵數量
    n_features = model.steps[0][1].n_output_features_
    print(f"[{name}] 特徵數: {n_features}, Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")

print("-" * 30)

# --- 4. 視覺化結果 (Visualization) ---

# 圖表 1: 預測效能比較
results_df = pd.DataFrame(results).T
print("\n效能比較表:")
print(results_df)

plt.figure(figsize=(10, 5))
results_df[['Train R2', 'Test R2']].plot(kind='bar')
plt.title('Bias-Variance Tradeoff: Train vs Test R2 Score')
plt.ylabel('R2 Score')
plt.ylim(-1, 1.2) # 限制 Y 軸範圍，因為 Linear 可能會負分到天邊
plt.axhline(0, color='black', linewidth=0.8)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 圖表 2: L1 vs L2 係數分佈 (觀察稀疏性)
plt.figure(figsize=(12, 6))

# 繪製 Ridge 係數
plt.plot(coefficients["Ridge (L2)"], 'o', label='Ridge (L2)', alpha=0.8, markersize=5)
# 繪製 Lasso 係數
plt.plot(coefficients["Lasso (L1)"], '^', label='Lasso (L1)', alpha=0.9, markersize=5)

plt.title('Coefficient Magnitude: Ridge (L2) vs Lasso (L1)')
plt.xlabel('Feature Index (0 to ~160)')
plt.ylabel('Coefficient Value (Weight)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 加入文字註解
plt.text(0, -0.5, "Notice how Lasso (Red Triangles) pushes many weights exactly to 0,\nwhile Ridge (Blue Dots) keeps them small but non-zero.", 
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()