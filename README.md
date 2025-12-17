# 30-Days-Of-ML

2. py -3.10 California_Housing_Prediction.py
3. py -3.10 Regularization_Demo.py
4. py -3.10 Logistic_Titanic.py
5. py -3.10 NaiveBayes_Spam.py

# 30 天 AI/ML 實戰挑戰：從入門到落地 (Story-Driven Edition)

這份課表採用 **「故事線 (Storyline)」** 設計，透過共用經典資料集（如鐵達尼號、員工離職數據），深入比較不同演算法在解決同一問題時的差異與進化。

---

| Day | 類別 | 演算法/主題 | 資料集 | 學習重點 |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **概論** | **AI 概論與地圖** | *(無)* | **起點**：AI vs ML vs DL 的關係、Python 環境建置 (Anaconda/Colab)。 |
| **02** | 回歸 | 線性回歸 (Linear Regression) | 薪資與年資數據 | $y = ax + b$ 的直覺、損失函數 (MSE) 的意義。 |
| **03** | 回歸 | 多元線性回歸 | 波士頓房價 (Boston Housing) | 處理多個特徵 (Features)、理解係數權重。 |
| **04** | 回歸 | 正則化 (Lasso / Ridge) | 加州房價預測 | 防止過擬合 (Overfitting)、L1 與 L2 的差別。 |
| **05** | 回歸 | 回歸模型評估 | *(沿用房價數據)* | 怎麼知道準不準？MAE, MSE, RMSE, $R^2$ Score。 |
| **06** | 監督 | 邏輯回歸 (Logistic Regression) | 糖尿病診斷 (Pima Diabetes) | 雖然叫回歸但是做分類、Sigmoid 函數、機率輸出。 |
| **07** | 監督 | K-近鄰 (KNN) | 鳶尾花分類 (Iris) | 距離計算 (歐式距離)、懶惰學習法 (Lazy Learning)。 |
| **08** | 監督 | 樸素貝氏 (Naive Bayes) | 垃圾郵件過濾 (Spam) | 條件機率、貝氏定理、處理文字特徵。 |
| **09** | 監督 | 決策樹 (Decision Tree) | 鐵達尼號生存 (Titanic) | 像人類思考的 If-Then 規則、資訊增益 (Entropy/Gini)。 |
| **10** | 監督 | 支持向量機 (SVM) | 乳癌檢測 (Breast Cancer) | 畫出最寬的「馬路」區分數據、核函數 (Kernel Trick)。 |
| **11** | 監督 | 數據前處理 (Preprocessing) | 鐵達尼號生存 (Titanic) | **重要地基**：缺失值處理、One-Hot Encoding、標準化。 |
| **12** | 監督 | 分類模型評估 | *(沿用鐵達尼號)* | **關鍵陷阱**：混淆矩陣、Precision vs Recall、F1-Score、ROC/AUC。 |
| **13** | 非監督 | K-Means 聚類 | 商場客戶分群 (Mall Customers) | 找出群聚中心、手肘法 (Elbow Method) 決定分幾群。 |
| **14** | 非監督 | 層次聚類 (Hierarchical) | 脊椎動物分類 | 樹狀圖 (Dendrogram)、由下而上的合併策略。 |
| **15** | 非監督 | DBSCAN | 地理位置數據 / 月亮數據 | 基於密度的聚類、處理噪聲與不規則形狀。 |
| **16** | 非監督 | 主成分分析 (PCA) | 葡萄酒分類 (Wine) | 降維神器、壓縮數據特徵、視覺化高維數據。 |
| **17** | 非監督 | 關聯規則 (Apriori) | 超市購物籃 (Market Basket) | 啤酒與尿布效應、支持度 (Support) 與信賴度 (Confidence)。 |
| **18** | 集成 | 隨機森林 (Random Forest) | 信用貸款風險 | **Bagging**：三個臭皮匠勝過諸葛亮、特徵重要性分析。 |
| **19** | 集成 | AdaBoost | 電信客戶流失 (Telco Churn) | **Boosting**：專注於修正前一個模型的錯誤。 |
| **20** | 集成 | **XGBoost** (競賽神器) | 保險索賠預測 | 梯度提升、正則化防止過擬合、Kaggle 必備技能。 |
| **21** | 集成 | LightGBM | 大型零售銷售數據 | 處理大數據更快、Leaf-wise 生長策略。 |
| **22** | 集成 | 模型調參 (Hyperparameter) | *(沿用 XGBoost 案例)* | Grid Search 與 Random Search、尋找最佳參數組合。 |
| **23** | 深度學習 | 感知機與 MLP | XOR 邏輯閘 / 手寫數字 | 神經元結構、全連接層 (Dense Layer)、激活函數。 |
| **24** | 深度學習 | 神經網路訓練機制 | MNIST 手寫數字 | 反向傳播 (Backpropagation)、優化器 (Adam/SGD)。 |
| **25** | 深度學習 | 卷積神經網路 (CNN) 基礎 | 貓狗圖片分類 | 卷積層 (特徵提取) 與池化層 (壓縮)。 |
| **26** | 深度學習 | 遷移學習 (Transfer Learning) | 花卉分類 (VGG16/ResNet) | 站在巨人肩膀上：使用預訓練模型解決小數據問題。 |
| **27** | 深度學習 | 循環神經網路 (RNN/LSTM) | 股價趨勢預測 | 處理時間序列、解決長短期記憶問題。 |
| **28** | 深度學習 | NLP 基礎 (Word Embeddings) | 電影評論情感分析 | 電腦如何讀懂字？Word2Vec 與 Tokenization。 |
| **29** | 深度學習 | 模型部署 (Deployment) | *(任選一個模型)* | 使用 Streamlit 快速將模型變成 Web App 展示。 |
| **30** | **總結** | **AI 總結與未來** | *(無)* | **終點即起點**：回顧這 29 天的技能樹、下一步建議。 |