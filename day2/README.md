## 0. 歷史小故事：回歸 (Regression) 的由來
「回歸」這個詞最早由英國科學家 **法蘭西斯·高爾頓爵士 (Sir Francis Galton)** 於 19 世紀末提出。作為達爾文的表弟，高爾頓熱衷於研究遺傳與生物統計，他最著名的研究之一就是觀察「父母身高與子女身高」的關係。

高爾頓發現了一個有趣的現象：雖然高個子的父母傾向生出高個子的子女，但這些子女的平均身高通常會**比父母矮一些**；反之，矮個子的父母所生的子女，平均身高則會**比父母高一些**。也就是說，極端的特徵在下一代會傾向「縮減」並向整體的平均值靠攏。

他將這種現象稱為 **「回歸平均值」 (Regression toward the Mean)**。當時的 "Regression" 原意是指生物特徵在遺傳過程中「倒退」回平均狀態的趨勢，而非我們現在所理解的「數據預測」。

隨著時間推移，統計學家（如 Pearson 和 Fisher）將高爾頓的概念數學化，發展出我們今日使用的線性模型。雖然現代的 **線性回歸 (Linear Regression)** 已經不再僅限於描述「回歸平均」的生物現象，而是指「利用自變數 $X$ 來預測應變數 $Y$ 的統計方法」，但這個充滿歷史意義的名字卻一直沿用至今。

## 1. 理論基礎 (Theory)
線性回歸 (Linear Regression) 是機器學習中最基礎的模型，也是許多複雜算法的基石。它的核心思想是尋找一條直線（或多維空間中的超平面），來擬合數據的分布趨勢，藉此預測連續型的數值。

### 1.1 核心公式
模型假設輸入特徵 `X` 與輸出 `y` 之間存在線性關係：

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

* **`ŷ` (Prediction)：** 模型的預測值。
* **`x` (Features)：** 輸入的特徵（例如：房屋的坪數、房齡）。
* **`w` (Weights/Coefficients)：** 權重。代表該特徵對結果的影響力。
    * 若 `w > 0`：正相關（例如：坪數越大，房價越高）。
    * 若 `w < 0`：負相關（例如：犯罪率越高，房價越低）。
* **`b` (Bias/Intercept)：** 截距項。代表當所有特徵為 0 時的基礎數值。

### 1.2 損失函數 (Loss Function)
我們的目標是讓預測值 `ŷ` 盡可能接近真實值 `y`。在線性回歸中，最常用的損失函數是 **均方誤差 (Mean Squared Error, MSE)**：

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

* 總樣本數：20,640 筆
* 特徵數：8 個 + 1 個目標變數

#### 模型效能：
* MSE (均方誤差): 0.5559
* R² Score (決定係數): 0.5758 (約 57.58%)

#### 特徵重要性排序：
* AveBedrms (平均臥室數): +0.783 ⬆️
* MedInc (收入中位數): +0.449 ⬆️
* HouseAge (房齡): +0.010 ⬆️
* Population (人口): -0.000002 ≈ 0
* AveOccup (平均居住人數): -0.004 ⬇️
* AveRooms (平均房間數): -0.123 ⬇️
* Latitude (緯度): -0.420 ⬇️
* Longitude (經度): -0.434 ⬇️

#### 圖表視窗：

* 收入中位數 vs 房價的散點圖

![收入中位數 vs 房價的散點圖](https://ithelp.ithome.com.tw/upload/images/20251210/20161788px5qCtT6PO.jpg)

* 真實值 vs 預測值的對比圖

![真實值 vs 預測值的對比圖](https://ithelp.ithome.com.tw/upload/images/20251210/20161788Vu9D0e7kfH.jpg)

* 房價分佈圖 (觀察天花板效應)
![房價分佈圖](https://ithelp.ithome.com.tw/upload/images/20251210/20161788TWJ5C8U9eU.jpg)

### 4. 實作心得與重點
* **MSE (均方誤差)**： 此數值越低越好，代表預測誤差越小。
本次實作的 MSE 為 `0.5559`。單看這個數字很難有感覺，我們需要結合資料單位的背景來解讀：

  1.  **單位換算：** 加州房價資料集的目標值 (`Target`) 單位是 **「10 萬美元 ($100,000)」**。
  2.  **MSE (均方誤差)：** 是誤差的平方，單位變成了「(10萬美元)²」，這不直觀。
  3.  **RMSE (均方根誤差)：** 將 MSE 開根號 `√0.5559 ≈ 0.745`。這代表單位的量級回到了「10 萬美元」。
  4.  **實際意義：** `0.745 × 100,000 = 74,500`。
    > **結論：** 這代表我們的模型預測房價時，**平均誤差約為 74,500 美元**。考慮到加州房價動輒數十萬美元，這個誤差範圍顯示簡單線性回歸只能捕捉大趨勢，精確度仍有提升空間。
    
* **R2 Score** (決定係數)： 最大值為 1。本次實作結果約在 0.57~0.60 之間，代表模型解釋了約 60% 的房價變異，這在僅使用簡單線性回歸的情況下是可以接受的基準點 (Baseline)。

* **權重意義**： 從係數中可以發現 MedInc (收入) 對房價有極強的正向影響，這符合經濟學直覺。


## 1.4 適用與不適用場景 (When to use?)

並非所有的預測問題都適合丟進線性回歸，了解它的極限與強項是資料科學家的第一課。

### ✅ 什麼時候適合用？ (Suitable Scenarios)

1.  **預測目標是「連續數值」 (Continuous Values)**
    * 你的 $y$ (Label) 必須是有意義的大小數值。
    * **例子：** 房價 (100萬, 200萬)、氣溫 (25.5度, 30度)、銷售量 (500件)、身高、體重。
    * **口訣：** 只要問題是問「多少 (How much/How many)?」，通常是回歸問題。

2.  **特徵與目標大致呈「線性關係」**
    * 當 $x$ 增加，$y$ 也大致依比例增加或減少。
    * 

3.  **資料量級別 (Data Size)**
    * **小數據的首選：** 線性回歸結構簡單，即使只有 **幾十筆 (e.g., 50筆)** 資料，也能畫出一條參考線。相較之下，深度學習 (Deep Learning) 在這種資料量下完全無法運作。
    * **OLS 極限：** 如果使用標準的最小平方法 (Ordinary Least Squares)，建議資料量在 **10 萬筆以下**。
        * 因為 OLS 涉及矩陣運算，計算複雜度較高，當資料超過 10 萬筆或特徵極多時，記憶體可能會爆掉或運算過久（此時需改用 SGDRegressor）。

### ❌ 什麼時候**不**適合？ (Unsuitable Scenarios)

1.  **分類問題 (Classification)**
    * 如果你要預測的是「類別」而非數值。
    * **例子：** 預測這封信是「垃圾郵件」還是「正常郵件」、預測病人「有得癌症」或「沒得癌症」。
    * 雖然可以用 0 和 1 代表，但線性回歸的預測值可能會跑出 1.5 或 -0.2 這種無意義的數字，這時應改用 **Day 02 的邏輯回歸 (Logistic Regression)**。

2.  **非線性關係嚴重 (Non-linear Relationship)**
    * 資料分佈呈現「曲線」、「拋物線」或「圓形」。
    * 
    * 如果你硬用一條直線去切拋物線資料，誤差會非常大（Underfitting）。

3.  **極端值 (Outliers) 太多**
    * 線性回歸非常容易被「離群值」拉走。
    * 舉例：全班平均資產，如果突然加入一個比爾蓋茲，那條「回歸線」會瞬間被拉高，導致對大部分普通人的預測都不準。

## 5. 實作反思：為什麼這個資料集很重要？ (Why this dataset?)

你可能會發現，即便我們跑完了線性回歸，R² 分數大約也只落在 **0.60** 上下。這是否代表模型很爛？其實不然，這個資料集在教育上有三個重要的意義：

### 1. 建立「基準線」 (Baseline) 的觀念
在機器學習中，線性回歸通常不是用來「贏得比賽」的，而是用來**「設立低標」**的。
* 它告訴我們：**「只用最簡單的直線，我們就能解釋 60% 的房價變化。」**
* 未來的 29 天，當你使用更強大的模型（如 XGBoost 或神經網路）時，你必須超過 0.60 才有意義。如果一個複雜的深度學習模型跑出 0.61，那說明為了提升 1% 的準度而犧牲運算資源是不值得的。

### 2. 相關性不等於因果性 (Correlation vs. Causation)
透過觀察權重係數 (Coefficients)，我們學到了**「可解釋性 (Interpretability)」**：
* **MedInc (收入中位數)** 的權重最高，這符合社會常識：有錢人住的地方房價高。
* **Latitude/Longitude (經緯度)** 的權重告訴我們：地點 (Location) 是決定房價的關鍵。
* 這比單純預測出一個數字更有商業價值。

### 3. 真實數據的「髒亂」 (Real-world Noise)
這個資料集包含了一個著名的資料陷阱：**「天花板效應 (Capped Values)」**。
如果你畫出房價的分布圖，會發現 50 萬美金（5.0）的地方有一根長長的柱子。這是因為當年普查時，為了統計方便，將所有超過 50 萬的房價都直接記為 50 萬。
* **教育意義：** 真實世界的數據永遠充滿了這種人為的誤差與限制。線性回歸在處理這種非線性截斷時會遇到困難，這正是為什麼我們需要更高級模型的原因。

## 5. 模型診斷：吳恩達的「火箭」哲學 (Model Diagnosis)

AI 專家吳恩達 (Andrew Ng) 曾提出著名的 **「火箭與燃料」** 比喻，這能幫助我們判斷目前的模型遇到什麼瓶頸，以及下一步該如何優化。



### 5.1 核心概念
* **火箭引擎 (The Engine) = 模型複雜度 (Model Complexity)**
    * 例如：線性回歸是小引擎，深度神經網路是巨大的引擎。
* **燃料 (The Fuel) = 數據量 (Data)**
    * 再強的引擎，沒有燃料也飛不起來；但如果是小引擎，給再多燃料也飛不快。

### 5.2 診斷流程 (The Workflow)
我們依序觀察 **訓練集 (Train)** 與 **測試集 (Test)** 的誤差表現：

1.  **第一關：檢查訓練集表現 (Train Error)**
    * **狀況：** 如果連訓練集的預測準確度都很差（High Bias / Underfitting）。
    * **診斷：** 引擎太小了！模型太簡單，無法捕捉數據的特徵。
    * **解法：** **更換更複雜的模型**（例如：從線性回歸 $\rightarrow$ 神經網路），或是增加特徵。
    * *（此時增加數據量通常沒用，因為小引擎已經極限了）*

2.  **第二關：檢查測試集表現 (Test Error)**
    * **狀況：** 訓練集很準，但測試集很差（High Variance / Overfitting）。
    * **診斷：** 引擎夠強，但沒油了（或是模型硬記答案）。
    * **解法：** **收集更多數據 (More Fuel)**，或是使用正規化 (Regularization) 限制模型。

3.  **第三關：表現皆良好**
    * **結論：** 訓練結束，準備部署。

### 4.3 回頭看本次實作 (Application)
在本次加州房價預測中，我們的線性回歸模型在訓練集與測試集的 $R^2$ 大約都在 **0.60** 左右。

* **現象：** 訓練集與測試集的分數**都不高**。
* **判定：** 這屬於 **High Bias (低度擬合)**。
* **結論：** 我們的問題不在於資料太少（兩萬筆夠了），而在於**「線性回歸這個引擎太小了」**。
* **Next Step：** 這也是為什麼我們在接下來的 30 天挑戰中，需要學習決策樹、XGBoost 甚至神經網路等「更大顆引擎」的原因。

---
Next Day Preivew: Day 03 - 正則化回歸 (Ridge & Lasso Regression)，重點： L1 與 L2 Regularization 的差別、如何解決 Overfitting、特徵篩選。