# 30天 AI/ML 演算法學習地圖 (Final Version)

| Day | 類別 | 演算法/主題 | 資料集 | 學習重點 |
| :--- | :--- | :--- | :--- | :--- |
| **01** | 概論 | AI 概論與地圖 | *(無)* | AI vs ML vs DL 的關係圖。 |
| **02** | 回歸 | 線性回歸 | 薪資與年資數據 | $y = ax + b$ 直覺、損失函數 (MSE)、梯度下降基礎。 |
| **03** | 回歸 | 多元線性回歸 | 創業公司利潤 | 多特徵處理、虛擬變數陷阱 (Dummy Variable Trap)。 |
| **04** | 回歸 | 正則化 (L1/L2) | 加州房價預測 | 關鍵概念：Bias vs Variance、防止過擬合。 |
| **05** | 基礎 | 數據預處理 | 鐵達尼號 | 地基工程：缺失值填補、One-Hot Encoding、特徵縮放 (Standardization)。 |
| **06** | 監督 | 邏輯回歸 | 社交網絡廣告 | 雖然叫回歸但是做分類、Sigmoid 函數、決策邊界。 |
| **07** | 監督 | K-近鄰 (KNN) | 鳶尾花分類 | 歐式距離計算、K 值選擇、數據縮放。 |
| **08** | 監督 | 樸素貝氏 | 垃圾郵件過濾 | 貝氏定理應用、NLP 基礎 (Bag of Words)。 |
| **09** | 監督 | 決策樹 | 鐵達尼號 | 視覺化決策樹、資訊增益 (Entropy)、基尼係數 (Gini)。 |
| **10** | 監督 | 支持向量機 (SVM) | 乳癌檢測 | 尋找最佳超平面、核函數 (Kernel Trick) 處理非線性。 |
| **11** | 進階 | 特徵工程 (Feature Eng) | *(綜合案例)* | 提升準度關鍵：特徵選擇 (Selection)、處理類別不平衡 (SMOTE)。 |
| **12** | 監督 | 分類模型評估 | *(沿用鐵達尼)* | 看懂成績單：混淆矩陣、Precision/Recall、F1-Score、ROC/AUC。 |
| **13** | 非監督 | K-Means 聚類 | 商場客戶分群 | 幾何中心迭代、手肘法 (Elbow Method) 決定 K 值。 |
| **14** | 非監督 | 層次聚類 | 脊椎動物分類 | 樹狀圖 (Dendrogram)、聚合式 (Bottom-up) 策略。 |
| **15** | 非監督 | DBSCAN | 月亮形狀數據 | 基於密度的聚類、解決不規則形狀與噪聲點問題。 |
| **16** | 非監督 | 主成分分析 (PCA) | 葡萄酒分類 | 降維神器：特徵壓縮、解釋變異量、高維數據視覺化。 |
| **17** | 非監督 | 關聯規則 (Apriori) | 超市購物籃 | 啤酒與尿布效應、支持度 (Support) 與信賴度 (Confidence)。 |
| **18** | 集成 | 隨機森林 (Random Forest) | 信用貸款風險 | Bagging：多棵樹投票機制、OOB Error、特徵重要性分析。 |
| **19** | 集成 | AdaBoost | 電信客戶流失 | Boosting：權重調整機制、弱分類器變強分類器。 |
| **20** | 集成 | XGBoost | 保險索賠預測 | 競賽神器：梯度提升樹、正則化目標函數、處理缺失值能力。 |
| **21** | 集成 | LightGBM | 大型零售銷售 | Leaf-wise 生長策略、處理大數據的高效能方案。 |
| **22** | 集成 | 模型調參 (Tuning) | *(沿用 XGBoost)* | Grid Search vs Random Search、貝葉斯優化觀念。 |
| **23** | Deep Learning | 感知機與 MLP | XOR 邏輯閘 | 神經元結構、全連接層 (Dense)、激活函數 (ReLU/Sigmoid)。 |
| **24** | Deep Learning | 神經網路訓練 | MNIST 手寫數字 | 反向傳播 (Backpropagation)、優化器 (Adam)。 |
| **25** | Deep Learning | CNN (卷積神經網路) | 貓狗圖片分類 | 卷積層 (特徵提取)、池化層 (降維)、Flatten。 |
| **26** | Deep Learning | 遷移學習 (Transfer) | 花卉分類 | 站在巨人肩膀上：使用 VGG16/ResNet 預訓練模型。 |
| **27** | Deep Learning | RNN (循環神經網路) | 模擬股價資料 | 時間序列基礎：處理序列數據、隱藏狀態 (Hidden State)。 |
| **28** | Deep Learning | LSTM (長短期記憶) | 模擬股價資料 | 解決梯度消失：Cell State、遺忘門/輸入門/輸出門。 |
| **29** | 應用 | 模型部署 (Deployment) | *(任選一個模型)* | 落地應用：使用 Streamlit 打造你的第一個 ML Web App。 |
| **30** | 總結 | AI 總結與未來 | *(無)* | 回顧與展望：技能樹總結、持續學習資源 (Paper/Kaggle)。 |
| **31** | NLP | Transformer 與 BERT | *(無)* | NLP 王者：Self-Attention 機制、BERT 實作情緒分析。 |
| **32** | CV | 物件偵測 (YOLO) | *(無)* | 電腦視覺進階：從分類到偵測、YOLO 原理與實作。 |
| **33** | GenAI | 生成式 AI (GAN) | 手寫數字 (MNIST) | 對抗生成網路：Generator vs Discriminator 的博弈。 |
| **34** | GenAI | 生成式 AI (VAE) | 手寫數字 (MNIST) | 變分自編碼器：Latent Space 的機率分佈與重構。 |
| **35** | GenAI | 生成式 AI (Diffusion) | 手寫數字 (MNIST) | 擴散模型：從雜訊中還原圖像 (Denoising)。 |
| **36** | RL | 強化學習 (Q-Learning) | 尋寶遊戲 | 價值基礎 (Value-Based)：Q-Table、Epsilon-Greedy 策略。 |
| **37** | RL | 強化學習 (DQN) | 倒立擺 (CartPole) | 深度 Q 網路：Experience Replay、Target Network。 |
| **38** | RL | 強化學習 (Policy Gradient) | 登陸月球 (LunarLander) | 策略基礎 (Policy-Based)：直接學習動作機率 (REINFORCE)。 |
| **39** | XAI | 可解釋 AI (XAI) | 房價預測 | 打開黑盒子：SHAP 值 (賽局理論)、LIME (局部解釋)。 |
| **40** | RAG | 檢索增強生成 (RAG) | 私有知識庫 | 讓 AI 讀懂資料：Embedding、Vector Search、LLM 生成。 |
| **41** | IPAS | IPAS 考前總複習 | 114年試題 | 考題分佈索引與綜合觀念解析。 |
| **42** | MLOps | MLOps 進階 | 模擬日誌資料 | Kubernetes, Drift Detection, CI/CD。 |
| **43** | Security | AI 安全與隱私 | *(無)* | 對抗式攻擊, 同態加密, 聯邦學習。 |
| **44** | Stats | 經典統計與時間序列 | 模擬銷售數據 | ARIMA, 統計檢定, 實驗設計。 |
