# Project Rules for 30-Days-Of-ML

> **CRITICAL RULE**: Do NOT automatically proceed to the next day's content without explicit user instruction. Always ask for confirmation before starting a new day.

> **CRITICAL RULE**: py -3.10

# 30天 AI/ML 演算法學習地圖 (Final Version)

| Day | 類別 | 演算法/主題 | 資料集 | 學習重點 |
| :--- | :--- | :--- | :--- | :--- |
| **01** | 概論 | AI 概論與地圖 | *(無)* | AI vs ML vs DL 的關係圖。 |
| **02** | 回歸 | 線性回歸 | 薪資與年資數據 | $y = ax + b$ 直覺、損失函數 (MSE)、梯度下降 (SGD) 基礎。 |
| **03** | 回歸 | 多元線性回歸 | 創業公司利潤 | 多特徵處理、虛擬變數陷阱 (Dummy Variable Trap)。 |
| **04** | 回歸 | 正則化 (L1/L2) | 加州房價預測 | 關鍵概念：Bias vs Variance、防止過擬合。 |
| **05** | 基礎 | 數據預處理 | 鐵達尼號 | 地基工程：缺失值填補、One-Hot Encoding、特徵縮放 (Standardization)。 |
| **06** | 監督 | 邏輯回歸 | 社交網絡廣告 | 雖然叫回歸但是做分類、Sigmoid 函數、決策邊界。 |
| **07** | 監督 | K-近鄰 (KNN) | 鳶尾花分類 | 歐式距離計算、K 值選擇、數據縮放。 |
| **08** | 監督 | 樸素貝氏 | 垃圾郵件過濾 | 貝氏定理應用、NLP 基礎 (CountVectorizer, Stopwords)。 |
| **09** | 監督 | 決策樹 | 鐵達尼號 | 視覺化決策樹、資訊增益 (Entropy)、基尼係數 (Gini)。 |
| **10** | 監督 | 支持向量機 (SVM) | 乳癌檢測 | 尋找最佳超平面、核函數 (Kernel Trick) 處理非線性。 |
| **11** | 進階 | 特徵工程 (Feature Eng) | *(綜合案例)* | 提升準度關鍵：特徵選擇 (Selection)、處理類別不平衡 (SMOTE)。 |
| **12** | 監督 | 分類模型評估 | *(沿用鐵達尼)* | 看懂成績單：混淆矩陣、F1-Score、ROC/AUC、交叉驗證 (K-Fold CV)。 |
| **13** | 非監督 | K-Means 聚類 | 商場客戶分群 | 幾何中心迭代、手肘法 (Elbow Method) 決定 K 值。 |
| **14** | 非監督 | 層次聚類 | 脊椎動物分類 | 樹狀圖 (Dendrogram)、聚合式 (Bottom-up) 策略。 |
| **15** | 非監督 | DBSCAN | 月亮形狀數據 | 基於密度的聚類、解決不規則形狀與噪聲點問題。 |
| **16** | 非監督 | 主成分分析 (PCA) | 葡萄酒分類 | 降維神器：特徵壓縮、解釋變異量、高維數據視覺化。 |
| **17** | 非監督 | 推薦系統 | 電影評分數據 | 協同過濾 (Collaborative Filtering)、矩陣分解 (SVD)。 |
| **18** | 集成 | 隨機森林 (Random Forest) | 信用貸款風險 | Bagging：多棵樹投票機制、OOB Error、特徵重要性分析。 |
| **19** | 集成 | AdaBoost | 電信客戶流失 | Boosting：權重調整機制、弱分類器變強分類器。 |
| **20** | 集成 | XGBoost | 保險索賠預測 | 競賽神器：梯度提升樹、正則化目標函數、處理缺失值能力。 |
| **21** | 集成 | LightGBM | 大型零售銷售 | Leaf-wise 生長策略、處理大數據的高效能方案。 |
| **22** | 集成 | 模型調參 (Tuning) | *(沿用 XGBoost)* | Grid Search vs Random Search、貝葉斯優化觀念。 |
| **23** | Deep Learning | 感知機與 MLP | XOR 邏輯閘 | 神經元結構、全連接層 (Dense)、激活函數 (ReLU/Sigmoid)。 |
| **24** | Deep Learning | 神經網路訓練 | MNIST 手寫數字 | 反向傳播 (Backpropagation)、優化器 (Adam)。 |
| **25** | Deep Learning | CNN (卷積神經網路) | 貓狗圖片分類 | 卷積層 (特徵提取)、池化層 (降維)、Flatten。 |
| **26** | Deep Learning | 遷移學習 (Transfer) | 花卉分類 | 站在巨人肩膀上：使用 VGG16/ResNet 預訓練模型。 |
| **27** | Deep Learning | RNN/LSTM | 國際航班乘客數 | 時間序列：處理序列數據、長短期記憶網絡、解決梯度消失。 |
| **28** | Deep Learning | NLP 基礎 | 電影評論情感 | 文字轉數字：Word Embeddings (Word2Vec) 與 Tokenization。 |
| **29** | 應用 | 模型部署 (Deployment) | *(任選一個模型)* | 落地應用：使用 Streamlit 打造你的第一個 ML Web App。 |
| **30** | 總結 | AI 總結與未來 | *(無)* | 回顧與展望：技能樹總結、持續學習資源 (Paper/Kaggle)。 |

## 1. Project Context
這是一個為期 30 天的機器學習實戰挑戰專案，目標是從入門到落地。
每一天 (Day XX) 都會對應一個特定的機器學習演算法或主題。
程式碼應寫在對應的 `dayXX` 資料夾中。

## 2. Article Generation Format (文章撰寫架構)
當被要求撰寫某一天的教學文章 (README.md) 時，必須嚴格遵守以下架構：

```markdown
# Day XX: [標題]

## 0. 歷史小故事/核心貢獻者:
(列出貢獻者或相關歷史故事，若無則留白或省略)

## 1. 資料集來源
### 資料集來源：[名稱](連結)
> 備註：(選填，如資料來源說明)

### 資料集特色與欄位介紹:
(列點說明資料集特色，例如：極簡潔、強相關性、無缺失值等)

**欄位說明**：
*   **Feature Name (特徵 X)**: (說明)
*   **Target Name (目標 y)**: (說明)

### 資料清理
(說明資料清理的步驟)

## 2. 原理
### 核心公式與參數
(簡述該演算法的核心數學概念或關鍵參數)

## 3. 實戰
### Python 程式碼實作
完整程式連結：[Script_Name.py](連結)

## 4. 模型評估
(根據模型類型選擇對應的評估方式)

### 若為回歸模型 (Regression)
*   **指標數字**：
    *   **R-Squared (R2)**: (數值，解釋)
    *   **MSE**: (數值)
    *   **Intercept/Coefficient**: (數值，解釋意義)
*   **圖表**：
    *   **預測結果圖**：(說明)
    *   **殘差圖 (Residual Plot)**：(說明)

### 若為分類模型 (Classification)
*   **指標數字**：
    *   **Accuracy**: (數值)
    *   **Confusion Matrix**: (數值)
*   **圖表**：
    *   **決策邊界圖** 或 **Feature Importance**

### 若為非監督模型 (Unsupervised)
*   **指標數字**：
    *   **Silhouette Score**: (數值)
*   **圖表**：
    *   **分群視覺化圖** 或 **手肘法圖**

## 5. 戰略總結:模型訓練的火箭發射之旅

### (回歸與監督式學習適用day2-12)
引用大師-吳恩達教授的 Rocket 進行說明 Bias vs Variance：
![rocket](https://github.com/ksharry/30-Days-Of-ML/blob/main/day02/pic/2-4_Rocket.jpg?raw=true)
(圖表固定用2-4這個網址)

#### 5.1 流程一：推力不足，無法升空 (Underfitting 迴圈)
*   **設定**：(參數設定)
*   **第一關：訓練集表現好嗎？**
*   **第二關：測試集表現好嗎？**
*   **行動 (Action)**：...

#### 5.2 流程二：動力太強，失控亂飛 (Overfitting 迴圈)
*   **設定**：...
*   **第一關：訓練集表現好嗎？**
*   **第二關：測試集表現好嗎？**
*   **行動 (Action)**：...

#### 5.3 流程三：完美入軌 (The Sweet Spot)
*   **設定**：...
*   **第一關 & 第二關**：訓練集與測試集表現都很好。
*   **結果**：完成！

### (非監督式學習適用day13-17)
*   討論分群效果評估：分得太細 (Over-segmentation) vs 分得太粗 (Under-segmentation) 的權衡。

## 6. 總結
(該日學習重點總結)
```

## 3. Coding Standards (程式碼規範)
*   **Python Version**: 3.10
*   **Libraries**: pandas, numpy, matplotlib, seaborn, sklearn (依需求)
*   **Plotting**:
    *   使用 `matplotlib.pyplot` 和 `seaborn`。
    *   **必須**將圖片儲存到 `dayXX/pic/` 資料夾中，例如 `plt.savefig('pic/XX-1.png')`。
    *   圖表需包含 Title, Labels, Legend (若適用)。
*   **Structure**:
    1.  載入資料
    2.  數據觀察 (EDA)
    3.  資料分割與前處理 (Standardization is important!)
    4.  建立與訓練模型
    5.  模型評估
    6.  結果視覺化

## 4. Code Generation Templates (程式碼生成模版)
When generating code for specific days, follow the architecture defined in the corresponding templates:

*   **Template 1 (Regression)**: `template_1.py`
    *   **Applicable Days**: Day 02-05
    *   **Key Features**: Train/Test Split, StandardScaler, MSE/R2 metrics, Residual Plot.

*   **Template 2 (Classification)**: `template_2.py`
    *   **Applicable Days**: Day 06-12 (Logistic, KNN, SVM, Decision Tree), Day 18-22 (Random Forest, XGBoost)
    *   **Key Features**: Confusion Matrix, Accuracy, Decision Boundary or Feature Importance.

*   **Template 3 (Clustering)**: `template_3.py`
    *   **Applicable Days**: Day 13-17 (K-Means, DBSCAN, PCA)
    *   **Key Features**: No y (Target), Silhouette Score, Cluster Visualization (Scatter plot with hue=Cluster).

*   **Template 4 (Deep Learning)**: `template_4.py`
    *   **Applicable Days**: Day 23-27 (MLP, CNN, RNN)
    *   **Key Features**: TensorFlow/Keras, History Plot (Loss/Accuracy curve), Model Summary.
