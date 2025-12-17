# Day 09：XGBoost 與 LightGBM - 梯度提升雙雄對決

## 0. 歷史小百科：
在機器學習的競賽歷史中，有兩個名字如同神一般的存在。
* **XGBoost (eXtreme Gradient Boosting)**：由陳天奇 (Tianqi Chen) 於 2014 年開發。它以優異的運算速度和準確度橫掃了當時各大 Kaggle 比賽，一度成為「奪冠標配」。它的出現將梯度提升樹 (GBDT) 的效能推向了極致。
* **LightGBM (Light Gradient Boosting Machine)**：由微軟 (Microsoft) 團隊於 2017 年發布。隨著大數據時代來臨，XGBoost 在處理海量數據時顯得記憶體消耗過大。LightGBM 橫空出世，主打「輕量」與「極速」，在保持準確率的同時，訓練速度比 XGBoost 快上數倍。

今天，我們就用這兩大殺器來挑戰德國信用違約預測。

## 1. 資料集來源
**資料集來源**：[UCI Machine Learning Repository - German Credit Data](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
這份資料集是金融風控領域的「Hello World」，由漢堡大學 (University of Hamburg) 的 Hans Hofmann 教授於 1994 年捐贈。它包含 1,000 筆真實的貸款紀錄，包含以下關鍵欄位，每一個都是徵信審核的重點：
![German Credit Data](https://github.com/ksharry/30-Days-Of-ML/blob/main/day7/pic/7-1.jpg?raw=true)

## 2. 原理
這兩者都屬於 **Boosting (提升法)** 家族，也就是「三個臭皮匠，勝過一個諸葛亮」。它們不是像 Random Forest 那樣並行訓練很多樹然後投票，而是**序列訓練**：第二棵樹修正第一棵樹的錯誤，第三棵樹修正前兩棵的錯誤，以此類推。

### 核心公式與參數

雖然兩者都屬於 Boosting 家族，但優化細節略有不同。

#### 1. XGBoost (eXtreme Gradient Boosting)
*   **核心公式**：
    $$Obj(\theta) = L(\theta) + \Omega(\theta)$$
    XGBoost 的特點是在目標函數中加入了明顯的 **正則化項 $\Omega(\theta)$**，這讓它比傳統 GBDT 更不容易過擬合。
*   **長樹方式 (Level-wise)**：
    像蓋樓房一樣，一層蓋完才蓋下一層。這種方式比較穩健，不容易長歪，但運算速度較慢。
*   **關鍵參數**：
    *   `eta` (learning_rate)：學習率，控制每棵樹的貢獻度。
    *   `max_depth`：樹的最大深度，通常設 3~10。
    *   `subsample`：隨機抽樣比例，防止過擬合。

#### 2. LightGBM (Light Gradient Boosting Machine)
*   **核心技術**：
    使用了 **GOSS (基於梯度的單邊採樣)** 和 **EFB (互斥特徵捆綁)** 技術，大幅減少了運算量。
*   **長樹方式 (Leaf-wise)**：
    像藤蔓一樣，哪個葉子節點誤差大，就往哪裡拼命長。這種方式收斂速度極快，但如果沒限制好，很容易鑽牛角尖 (Overfitting)。
*   **關鍵參數**：
    *   `learning_rate`：學習率。
    *   `num_leaves`：葉子節點總數。這是 LightGBM 最重要的參數，通常 `num_leaves` < $2^{max\_depth}$。
    *   `min_data_in_leaf`：葉子最少資料數，用來防止過擬合的重要煞車。

## 3. 實戰：信用違約預測
### Python 程式碼實作
我們將同時實作這兩個模型，並比較它們的表現。
完整程式連結：[XGB_vs_LGBM.py](https://github.com/ksharry/30-Days-Of-ML/blob/main/day9/XGB_vs_LGBM.py)

## 4. 模型評估
我們在德國信用資料集上進行了測試，結果如下：

### 整體成績單
* 🔹 **XGBoost 測試集準確率 (Test Acc): 0.7733**
* 🔹 **LightGBM 測試集準確率 (Test Acc): 0.7800**

在此資料集上，XGBoost 略勝一籌，但兩者都優於昨天的隨機森林。LightGBM 的優勢在於如果資料量擴大到 100 萬筆，它的速度會輾壓 XGBoost。

### 圖解分析-特徵重要性
![feature_importance](https://github.com/ksharry/30-Days-Of-ML/blob/main/day9/pic/9-1.jpg?raw=true)
解讀：
*   **Checking Account (帳戶餘額)**：這通常是預測違約最強的指標。如果一個人的帳戶餘額是負的，或者根本沒有支票帳戶，違約風險會顯著上升。
*   **Credit Amount (貸款金額)**：貸款金額越大，還款壓力越大，風險自然越高。
*   **Duration (貸款期限)**：時間越長，變數越多，風險也隨之增加。
*   **Age (年齡)**：年輕人可能收入不穩，風險相對較高；而中年人通常財務狀況較穩定。

值得注意的是，XGBoost 和 LightGBM 對特徵重要性的計算方式略有不同（Weight vs Split），但它們通常都能抓出最關鍵的幾個影響因子。

### 圖解分析-混淆矩陣
![confusion_matrix](https://github.com/ksharry/30-Days-Of-ML/blob/main/day9/pic/9-2.jpg?raw=true)
解讀：
*   **準確率 (Accuracy)**：LightGBM (0.7800) 在此測試中略高於 XGBoost (0.7733)。
*   **關鍵取捨 (Trade-off)**：在銀行風控中，我們更在意的是 **False Negative (FN)**，也就是「預測為好人，結果他是壞人」。
    *   看圖中的左下角區塊：如果這個數字太高，代表銀行借錢給了會倒帳的人，這會造成直接的財務損失。
    *   雖然 LightGBM 整體準確率較高，但我們也要觀察它在「抓壞人」這件事上是否也同樣優秀。

## 5. 戰略總結:模型訓練的火箭發射之旅
最後，讓我們引用 AI 大師 **吳恩達 (Andrew Ng)** 的經典圖表，來重新審視我們學到的模型：
![rocket](https://github.com/ksharry/30-Days-Of-ML/blob/main/day2/pic/2-6.jpg?raw=true)

Boosting 模型就像威力強大的火箭引擎，參數設得好能飛天，設不好會直接爆炸。

### 5.1 流程一：推力不足，無法升空 (Underfitting 迴圈)
* **設定**：樹的數量太少 (e.g., `n_estimators=10`) 或樹太淺 (`max_depth=1`)。
* **第一關：訓練集表現好嗎？** ❌ 不好，模型連考古題都背不起來。
* **第二關：測試集表現好嗎？** ❌ 當然也不好。
* **行動 (Action)**：增加樹的數量 (`n_estimators`)，或增加樹的深度。

### 5.2 流程二：動力太強，失控亂飛 (Overfitting 迴圈)
* **設定**：樹長得太深 (`max_depth=20`) 或學習率太高 (`learning_rate=1.0`)。這在 Boosting 模型中最常見。
* **第一關：訓練集表現好嗎？** ✅ 完美！準確率 100%。
* **第二關：測試集表現好嗎？** ❌ 很差，模型「死背」了答案，不會舉一反三。
* **行動 (Action)**：
    1.  降低學習率 (`learning_rate`)。
    2.  限制樹深 (`max_depth`)。
    3.  增加正則化參數 (`reg_alpha`, `reg_lambda`)。

### 5.3 流程三：完美入軌 (The Sweet Spot)
* **設定**：適中的樹深 (3~6)，適中的學習率 (0.01~0.1)。
* **第一關 & 第二關**：
    * 訓練集與測試集表現都很好 (**Test Acc ≈ 77.6%**)。
* **結果**：**完成！** 這是目前的 SOTA (State of the Art) 等級表現。

## 6. 總結與比較
| 模型 | Day | 特性 | 德國信用資料準確率* |
| :--- | :---: | :--- | :---: |
| **決策樹** | 07 | 白箱模型，可畫圖解釋 | 70.00% |
| **隨機森林** | 08 | 多樹投票，穩定運行 | 76.00% |
| **XGBoost** | **09** | **精度之王，Kaggle 神器** | **77.33%** |
| **LightGBM** | **09** | **速度之王，海量數據首選** | **78.00%** |


**Next Day 預告：**
明天 Day 10，我們將進入機器學習的 **「可解釋性 (Explainability)」** 領域。
雖然 XGBoost 和 LightGBM 很準，但它們常被詬病為「黑盒子」。為什麼模型會拒絕這筆貸款？是因為收入太低？還是因為年齡太輕？
**SHAP (SHapley Additive exPlanations)** 將賦予我們透視黑盒子的能力，讓我們不僅知其然，更知其所以然！我們將學會如何精準計算每一個特徵對預測結果的貢獻度。