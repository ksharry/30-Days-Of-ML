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
完整程式連結-[California_Housing_Prediction.py](https://github.com/ksharry/30-Days-Of-ML/blob/main/day2/California_Housing_Prediction.py)

```python
# --- 2. 數據觀察 (EDA) ---
# 觀察「收入中位數 (MedInc)」與「房價 (Target)」的關係
plt.figure(figsize=(8, 6))
sns.scatterplot(x='MedInc', y='Target', data=df, alpha=0.3)
plt.title('Median Income vs House Value')
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

### 4 視覺化觀察 (Visual Analysis)
我們透過四張圖表來深入剖析模型的表現與資料的特性：

#### 4.1 特徵關係：收入 vs 房價
![Income vs House Value](https://ithelp.ithome.com.tw/upload/images/20251210/20161788px5qCtT6PO.jpg)
* **觀察：** `MedInc` (收入中位數) 與房價呈現明顯的 **正相關**，這解釋了為什麼它在模型中的權重最高。
* **細節：** 注意圖表上方有一條水平的橫線（與一些斷續的水平線），這預示了資料被「人為截斷」的痕跡。

#### 4.2 預測結果：真實值 vs 預測值
![True vs Predicted](https://ithelp.ithome.com.tw/upload/images/20251210/20161788Vu9D0e7kfH.jpg)
* **觀察：** 紅色虛線代表完美預測。點越靠近紅線越好。
* **異常：** 注意最右側 **X=5.0** 處，有一整排垂直的點。這代表對於那些真實價值 50 萬以上的豪宅，模型**嚴重低估**了它們的價格（因為模型沒看過大於 50 萬的數據，只能依照線性規律去猜）。

#### 4.3 地理空間分析
![Geospatial Map](https://ithelp.ithome.com.tw/upload/images/20251210/20161788dJ1YIF5RPt.jpg)
* **觀察：** 圖中顏色越黃代表房價越高，圓圈大小代表人口。
* **洞察：** 高房價清楚地集中在 **沿海地區**（舊金山灣區與洛杉磯）。
* **限制：** 線性回歸只能處理簡單的數值增減（例如：緯度越低越貴？），但無法理解這種「沿著海岸線分布」的複雜地理聚落，這也是模型分數無法突破的主因之一。

#### 4.4 殘差分析 (Residual Plot)
![Residual Plot](https://ithelp.ithome.com.tw/upload/images/20251210/20161788FNBCp58ZJD.jpg)
* **觀察：** 殘差 = 真實值 - 預測值。理想的殘差圖應該像一團隨機散亂的雲。
* **警訊：** 圖片右上角出現了一條明顯的 **切線邊界**。這是天花板效應留下的「疤痕」（因為 $True Value$ 被卡在 5.0，導致殘差呈現線性規律）。這再次證明線性模型無法完美處理這種非自然截斷的數據。

---

## 5. 深度反思與診斷 (Reflection & Diagnosis)

為什麼 $R^2$ 只有 0.58？我們可以用 AI 專家吳恩達 (Andrew Ng) 的 **「火箭與燃料」** 哲學來診斷目前的模型狀態。
![火箭升空翻譯](https://ithelp.ithome.com.tw/upload/images/20251210/201617881l3wPSfzsf.jpg)

### 5.1 診斷流程
我們依序觀察 **訓練集** 與 **測試集** 的表現：

1.  **第一關：檢查訓練集表現 (Train Error)**
    * **現象：** 分數不高 (0.6 左右)。
    * **診斷：** **Underfitting (低度擬合)**。這代表我們的「火箭引擎」（線性回歸）太小了，無法捕捉複雜的房價變化。
    * **解法：** 更換更複雜的模型（例如神經網路），或增加多項式特徵。

2.  **第二關：檢查測試集表現 (Test Error)**
    * **現象：** 與訓練集差不多差。
    * **結論：** 問題不在於資料不足（燃料夠），而在於模型太簡單。

### 5.2 總結
在本次實作中，我們屬於 **High Bias (高偏差)** 的情況。這告訴我們，單純增加數據量（燃料）已經沒有幫助，下一階段的挑戰，我們需要換上**「更大的引擎」**來突破 0.6 的天花板。

---
Next Day Preivew: Day 03 - 正則化回歸 (Ridge & Lasso Regression)，重點： L1 與 L2 Regularization 的差別、如何解決 Overfitting、特徵篩選。
