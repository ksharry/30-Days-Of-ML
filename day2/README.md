## 1. 理論基礎 (Theory)

線性回歸 (Linear Regression) 是機器學習中最基礎的模型，也是許多複雜算法的基石。它的核心思想是尋找一條直線（或多維空間中的超平面），來擬合數據的分布趨勢，藉此預測連續型的數值。

### 1.1 核心公式
模型假設輸入特徵 $X$ 與輸出 $y$ 之間存在線性關係：

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

* **$\hat{y}$ (Prediction)：** 模型的預測值。
* **$x$ (Features)：** 輸入的特徵（例如：房屋的坪數、房齡）。
* **$w$ (Weights/Coefficients)：** 權重。代表該特徵對結果的影響力。
    * 若 $w > 0$：正相關（例如：坪數越大，房價越高）。
    * 若 $w < 0$：負相關（例如：犯罪率越高，房價越低）。
* **$b$ (Bias/Intercept)：** 截距項。代表當所有特徵為 0 時的基礎數值。

### 1.2 損失函數 (Loss Function)
我們的目標是讓預測值 $\hat{y}$ 盡可能接近真實值 $y$。在線性回歸中，最常用的損失函數是 **均方誤差 (Mean Squared Error, MSE)**：

$$J(w, b) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

### 1.3 優化方法 (Optimization)
為了找到讓誤差最小的 $w$ 和 $b$，通常使用以下方法：
1.  **最小平方法 (Ordinary Least Squares, OLS)：** 透過數學解析解直接算出最佳參數（Scikit-Learn 的 `LinearRegression` 預設使用此法）。
2.  **梯度下降法 (Gradient Descent)：** 透過迭代更新權重來尋找最低點（適合極大規模數據）。

---

## 2. 實作：加州房價預測 (California Housing Prediction)

本實作使用 Scikit-Learn 內建的 [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)資料集。此資料集是取代舊版波士頓房價資料集（因倫理爭議被棄用）的標準入門數據。

### 2.1 資料集介紹
* **來源：** 1990 年加州人口普查數據。
* **樣本數：** 20,640 筆。
* **特徵 (Features)：** 8 個，包含 MedInc (收入中位數), HouseAge (房齡), AveRooms (平均房間數) 等。
* **目標 (Target)：** 該街區的房價中位數 (MedHouseVal)，單位為 10 萬美元。

![加州房價預測](https://ithelp.ithome.com.tw/upload/images/20251210/20161788uCq3qJWypk.jpg)

### 2.2 Python 程式碼實作

```python
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
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # 完美預測線
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted House Values')
plt.show()
```

### 3. 資料集資訊：

總樣本數：20,640 筆
特徵數：8 個 + 1 個目標變數
模型效能：

MSE (均方誤差): 0.5559
R² Score (決定係數): 0.5758 (約 57.58%)

特徵重要性排序：
AveBedrms (平均臥室數): +0.783 ⬆️
MedInc (收入中位數): +0.449 ⬆️
HouseAge (房齡): +0.010 ⬆️
Population (人口): -0.000002 ≈ 0
AveOccup (平均居住人數): -0.004 ⬇️
AveRooms (平均房間數): -0.123 ⬇️
Latitude (緯度): -0.420 ⬇️
Longitude (經度): -0.434 ⬇️

程式顯示了兩個圖表視窗：

收入中位數 vs 房價的散點圖
![收入中位數 vs 房價的散點圖](https://ithelp.ithome.com.tw/upload/images/20251210/20161788px5qCtT6PO.jpg)
真實值 vs 預測值的對比圖
![真實值 vs 預測值的對比圖](https://ithelp.ithome.com.tw/upload/images/20251210/20161788Vu9D0e7kfH.jpg)


### 4. 實作心得與重點
* **MSE (均方誤差)**： 此數值越低越好，代表預測誤差越小。

* **R2 Score** (決定係數)： 最大值為 1。本次實作結果約在 0.57~0.60 之間，代表模型解釋了約 60% 的房價變異，這在僅使用簡單線性回歸的情況下是可以接受的基準點 (Baseline)。

* **權重意義**： 從係數中可以發現 MedInc (收入) 對房價有極強的正向影響，這符合經濟學直覺。
---
Next Day Preivew: Day 03 - 正則化回歸 (Ridge & Lasso Regression)，重點： L1 與 L2 Regularization 的差別、如何解決 Overfitting、特徵篩選。