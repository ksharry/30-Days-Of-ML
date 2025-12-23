# Day 20: XGBoost (Extreme Gradient Boosting) - 糖尿病預測

## 0. 歷史小故事/核心貢獻者:
**XGBoost (eXtreme Gradient Boosting)** 由 **陳天奇 (Tianqi Chen)** 於 2014 年開發。
它一問世就橫掃了 Kaggle 各大比賽，被稱為「Kaggle 神器」。
它的核心精神是：**「天下武功，唯快不破」**。它在 Gradient Boosting 的基礎上，對系統效能做了極致的優化，讓訓練速度快了 10 倍以上，且準確度更高。

## 1. 資料集來源
### 資料集來源：[Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
> 備註：我們使用 Scikit-Learn 的 `fetch_openml` 下載。

### 資料集特色與欄位介紹:
這是一個經典的醫療數據集，用於預測皮馬印第安人是否患有糖尿病。
*   **目標 (Target)**：`class` (1=陽性/有糖尿病, 0=陰性/健康)。
*   **特徵 (Features)**：共 8 個，包含：
    *   `plas`: 血糖濃度 (最重要的指標)。
    *   `mass`: BMI 指數。
    *   `age`: 年齡。
    *   `preg`: 懷孕次數。
    *   `pedi`: 糖尿病家族函數 (遺傳風險)。
    *   `insu`: 胰島素濃度。
    *   `pres`: 血壓。
    *   `skin`: 皮膚皺褶厚度。

## 2. 原理
### 核心概念：改考卷的老師 (Gradient Boosting)

#### 2.1 梯度提升 (Gradient Boosting)
如果說 AdaBoost 是「錯題本複習」(加權重)，那 Gradient Boosting 就是 **「改考卷」**。
*   **第一棒 (Model 1)**：考了 60 分。
*   **殘差 (Residual)**：還差 40 分 (100 - 60)。
*   **第二棒 (Model 2)**：**它的目標不是考 100 分，而是去預測那「差的 40 分」**。假設它預測出了 30 分。
*   **目前總分**：60 + 30 = 90 分。還差 10 分。
*   **第三棒 (Model 3)**：去預測那「差的 10 分」。
*   **最後**：把所有人的分數加起來，就是最終預測。

#### 2.2 XGBoost 的「Extreme」在哪裡？
XGBoost 是 Gradient Boosting 的**超級進化版**：
1.  **正則化 (Regularization)**：它在公式裡加了懲罰項，防止模型「死記硬背」 (Overfitting)。這就像老師規定「解題步驟不能太複雜」，強迫學生學會通用的解法。
2.  **二階導數 (Second Order Derivative)**：傳統只用一階導數 (斜率) 找方向，XGBoost 用了二階 (曲率)，找得更準、更快。
3.  **系統優化**：支援平行運算 (Parallel Processing)，善用 CPU 的每一個核心，所以速度超快。

#### 2.3 國中生也能懂的案例：高爾夫球推桿
想像你在打高爾夫球，目標是把球打進洞 (預測準確)：
1.  **第一桿 (Model 1)**：用力一揮，球離洞口還差 100 公尺 (殘差)。
2.  **第二桿 (Model 2)**：你不需要再從起點打，你是從**球現在的位置** (100 公尺處) 往洞口推。你推了 80 公尺，還差 20 公尺。
3.  **第三桿 (Model 3)**：再輕輕推 18 公尺，還差 2 公尺。
4.  **最後**：幾次修正後，球就進洞了！

## 3. 實戰
### Python 程式碼實作
完整程式連結：[XGBoost_Diabetes.py](XGBoost_Diabetes.py)

```python
# 關鍵程式碼：XGBoost

# 1. 匯入 XGBoost
from xgboost import XGBClassifier

# 2. 訓練模型
# n_estimators=100: 最多打 100 桿
# learning_rate=0.1: 每一桿的力道 (太大力容易打過頭，太小力要打很久)
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

# 3. 邊訓練邊驗證 (Early Stopping)
# 如果打了 10 桿發現球都沒有更靠近洞口，就提早結束，省力氣
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
```

## 4. 模型評估與視覺化
### 1. 混淆矩陣 (Confusion Matrix)
![Confusion Matrix](pic/20-1_Confusion_Matrix.png)
*   **準確率 (Accuracy)**：約 **76.0%**。
*   **觀察**：
    *   對於沒有糖尿病 (0) 的預測較準 (Recall 0.80)。
    *   對於有糖尿病 (1) 的預測稍弱 (Recall 0.69)。這在醫療上是可以接受的起點，但通常我們希望 Recall 更高 (寧可誤判有病，也不要漏掉病人)。

### 2. 特徵重要性 (Feature Importance)
![Feature Importance](pic/20-2_Feature_Importance.png)
*   **觀察**：XGBoost 認為最重要的特徵是：
    1.  **plas (血糖濃度)**：毫無疑問，這是糖尿病最直接的指標。
    2.  **mass (BMI)**：肥胖是糖尿病的主要風險因子。
    3.  **age (年齡)**：年紀越大風險越高。
*   **價值**：這跟醫學常識完全吻合！證明模型真的學到了東西。

### 3. 學習曲線 (Learning Curve)
![Learning Curve](pic/20-3_Learning_Curve.png)
*   **觀察**：
    *   藍線 (Train Loss) 一路下降，代表模型一直在學習。
    *   橘線 (Test Loss) 一開始下降，但後來持平甚至微幅上升。
    *   這代表模型在後面可能開始有點 **Overfitting** (鑽牛角尖) 了。這時候 Early Stopping 就很有用，可以在橘線最低點時喊停。

## 5. 戰略總結: 集成學習的火箭發射之旅

### (XGBoost 適用)

#### 5.1 流程一：精準打擊 (Gradient Boosting)
*   **設定**：每一棵樹都專注於預測「上一棵樹的殘差」。
*   **結果**：誤差越來越小，逼近完美。

#### 5.2 流程二：極速狂飆 (System Optimization)
*   **設定**：使用 XGBoost 的平行運算和快取優化。
*   **結果**：訓練速度比傳統 GBDT 快 10 倍，能處理海量數據。

#### 5.3 流程三：自我約束 (Regularization)
*   **設定**：加入正則化項 (L1/L2)。
*   **結果**：模型不會為了考 100 分而死記硬背，泛化能力更強。

## 6. 總結
Day 20 我們學習了 **XGBoost**。
*   它是目前結構化數據 (表格資料) 的**最強模型**之一。
*   它結合了 **Gradient Boosting (不斷修正殘差)** 和 **系統優化 (速度快)** 的優點。
*   **高爾夫球推桿** 的比喻讓我們理解了它「逐步逼近目標」的原理。

下一章 (Day 21)，我們將進入 **Stacking (堆疊法)**，這是一種「集大成」的策略，把我們之前學過的 KNN, SVM, Random Forest, XGBoost 全部疊在一起，打造一個超級無敵的混合模型！
