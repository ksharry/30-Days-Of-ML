# Day 01：線性回歸 (Linear Regression) 與房價預測

## 0. 歷史小故事：回歸 (Regression) 的由來
「回歸」這個詞最早由英國科學家 **法蘭西斯·高爾頓爵士 (Sir Francis Galton)** 於 19 世紀末提出。作為達爾文的表弟，高爾頓熱衷於研究遺傳與生物統計，他最著名的研究之一就是觀察「父母身高與子女身高」的關係。



高爾頓發現了一個有趣的現象：雖然高個子的父母傾向生出高個子的子女，但這些子女的平均身高通常會**比父母矮一些**；反之，矮個子的父母所生的子女，平均身高則會**比父母高一些**。也就是說，極端的特徵在下一代會傾向「縮減」並向整體的平均值靠攏。

他將這種現象稱為 **「回歸平均值」 (Regression toward the Mean)**。當時的 "Regression" 原意是指生物特徵在遺傳過程中「倒退」回平均狀態的趨勢，而非我們現在所理解的「數據預測」。

隨著時間推移，統計學家（如 Pearson 和 Fisher）將高爾頓的概念數學化，發展出我們今日使用的線性模型。雖然現代的 **線性回歸 (Linear Regression)** 已經不再僅限於描述「回歸平均」的生物現象，而是指「利用自變數 $X$ 來預測應變數 $Y$ 的統計方法」，但這個充滿歷史意義的名字卻一直沿用至今。

---

## 1. 理論基礎 (Theory)
線性回歸 (Linear Regression) 是機器學習中最基礎的模型。它的核心思想是尋找一條直線（或多維空間中的超平面），來擬合數據的分布趨勢，藉此預測連續型的數值。

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

### 1.4 適用與不適用場景 (When to use?)

並非所有的預測問題都適合丟進線性回歸，了解它的極限與強項是資料科學家的第一課。

#### ✅ 什麼時候適合用？ (Suitable Scenarios)
1.  **預測目標是「連續數值」**：如房價、氣溫、銷售量。只要問題是問「多少 (How much)?」，通常是回歸問題。
2.  **特徵與目標大致呈「線性關係」**：特徵增加，目標也依比例增減。
3.  **小數據 (<10萬筆)**：線性回歸結構簡單，運算極快，是小數據集的首選 Baseline。

#### ❌ 什麼時候不適合？ (Unsuitable Scenarios)
1.  **分類問題 (Classification)**：如預測 Email 是垃圾郵件 (0/1)。
2.  **非線性關係嚴重**：資料呈現拋物線或圓形分佈。
3.  **極端值 (Outliers) 太多**：線性回歸容易被離群值（如億萬富翁的資產）拉走，導致整體預測失準。

---

## 2. 實作：加州房價預測 (California Housing Prediction)

本實作使用 Scikit-Learn 內建的 [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) 資料集。

### 2.1 資料集介紹
* **來源：** 1990 年加州人口普查數據。
* **樣本數：** 20,640 筆。
* **特徵 (Features)：** 8 個，包含 MedInc (收入中位數), HouseAge (房齡), AveRooms (平均房間數) 等。
* **目標 (Target)：** 該街區的房價中位數 (MedHouseVal)，單位為 **10 萬美元**。
![資料內容](https://ithelp.ithome.com.tw/upload/images/20251210/20161788oLGBSGPViW.jpg)

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

# --- 7. 結果視覺化 (觀察天花板效應) ---
plt.figure(figsize=(10, 6))
sns.histplot(df['Target'], bins=50, kde=True)
plt.axvline(x=5.0, color='red', linestyle='--', label='Ceiling ($500k)')
plt.title('Distribution of House Prices')
plt.legend()
plt.show()
```
## 3. 結果與分析 (Results & Analysis)

### 3.1 模型效能指標
* **MSE (均方誤差): `0.5559`**
    * **解讀：** 由於資料單位是「10 萬美元」，直接看 MSE 較無感。將其開根號 (RMSE) 約為 `0.745`。
    * **結論：** 這代表模型預測房價時，**平均誤差約為 74,500 美元**。考慮到加州房價動輒數十萬，這個誤差顯示模型抓到了大方向，但精確度仍有空間。
* **R² Score (決定係數): `0.5758` (約 57.6%)**
    * **解讀：** 模型解釋了約 58% 的房價變異。以社會科學數據（充滿雜訊）而言，這是一個及格的基準線 (Baseline)。

### 3.2 特徵重要性 (Feature Importance)
從係數 (`Coefficient`) 可以看出哪些因素最影響房價：
* **⬆️ 正相關最強：** `MedInc` (+0.449)。收入中位數越高，房價越高（符合直覺）。
* **⬇️ 負相關最強：** `Latitude` / `Longitude` (-0.42 / -0.43)。這顯示「地理位置」是決定房價的關鍵因素（例如越往內陸或特定緯度房價越低）。

### 3.3 視覺化觀察
* **天花板效應 (The Ceiling Effect)：**
    ![房價分佈圖](https://ithelp.ithome.com.tw/upload/images/20251210/20161788TWJ5C8U9eU.jpg)
    觀察上圖最右側，在 `5.0` (50萬美元) 處有一根異常高的柱子。這是因為當年的普查將所有超過 50 萬的房價都標記為 50 萬。這就是真實數據中的「人為截斷 (Censored Data)」，會影響線性回歸的準確度。

---

## 4. 深度反思與診斷 (Reflection & Diagnosis)

為什麼 $R^2$ 只有 0.58？我們可以用 AI 專家吳恩達 (Andrew Ng) 的 **「火箭與燃料」** 哲學來診斷目前的模型狀態。
![火箭升空解釋](https://ithelp.ithome.com.tw/upload/images/20251210/20161788MrCn7oI8mN.jpg)

### 4.1 診斷流程
我們依序觀察 **訓練集** 與 **測試集** 的表現：

1.  **第一關：檢查訓練集表現 (Train Error)**
    * **現象：** 分數不高 (0.6 左右)。
    * **診斷：** **Underfitting (低度擬合)**。這代表我們的「火箭引擎」（線性回歸）太小了，無法捕捉複雜的房價變化。
    * **解法：** 更換更複雜的模型（例如神經網路），或增加多項式特徵。

2.  **第二關：檢查測試集表現 (Test Error)**
    * **現象：** 與訓練集差不多差。
    * **結論：** 問題不在於資料不足（燃料夠），而在於模型太簡單。

### 4.2 總結
在本次實作中，我們屬於 **High Bias (高偏差)** 的情況。這告訴我們，單純增加數據量（燃料）已經沒有幫助，下一階段的挑戰，我們需要換上**「更大的引擎」**來突破 0.6 的天花板。

---
Next Day Preivew: Day 03 - 正則化回歸 (Ridge & Lasso Regression)，重點： L1 與 L2 Regularization 的差別、如何解決 Overfitting、特徵篩選。
