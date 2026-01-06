# Day 41: IPAS 中級能力鑑定 - 考前總複習 (Exam Review)

## 0. 前言：最後一哩路
恭喜你完成了前 40 天的實戰訓練！
現在，我們已經具備了通過 **IPAS 機器學習工程師 (中級)** 認證的實力。

為了幫助你順利考取證照，我們將 Day 41 設定為 **「考前總複習」**。
我們已經將大部分的分類考題分散到了各個章節 (Day 12, 25, 31, 32, 33...)。
本章節將作為一個 **索引 (Index)**，並補充尚未歸類的 **綜合性考題**。

## 1. 考題分佈索引
請依照以下地圖，回顧各個章節的重點考題：

| 類別 | 關鍵字 | 對應章節 |
| :--- | :--- | :--- |
| **模型評估** | F1-Score, ROC/AUC, 混淆矩陣, 交叉驗證 | [Day 12: 分類模型評估](../day12/README.md) |
| **CNN** | 卷積層, 池化層, VGG16, 資料增強 | [Day 25: CNN](../day25/README.md) |
| **NLP** | Transformer, BERT, Word2Vec, Attention | [Day 31: Transformer](../day31/README.md) |
| **物件偵測** | YOLO, mAP, IoU, NMS | [Day 32: YOLO](../day32/README.md) |
| **GenAI** | GAN, VAE, Diffusion, CLIP | [Day 33: GAN](../day33/README.md) |
| **MLOps** | Kubernetes, Drift Detection, CI/CD | [Day 42: MLOps](../day42/README.md) |
| **AI 安全** | 對抗式攻擊, 聯邦學習, 同態加密 | [Day 43: AI 安全](../day43/README.md) |

## 2. 綜合觀念與其他考題
以下彙整了跨領域或較難歸類的觀念題：

### 2.1 基礎數學與統計
| 題目 (關鍵字) | 答案 | 解析 |
| :--- | :--- | :--- |
| **(第三科) 40. 矩陣運算** | (C) `np.dot(v1, v2)` 為內積 | 這是 NumPy 最基礎的運算，也是神經網路的核心。 |
| **(第三科) 41. 條件機率** | (D) $P(A|B) = P(A \cap B) / P(B)$ | 貝氏定理的基礎公式。 |
| **(第三科) 15. R-Squared** | (B) 85% 的變異可被模型解釋 | $R^2$ 是迴歸模型最直觀的解釋力指標。 |

### 2.2 深度學習綜合
| 題目 (關鍵字) | 答案 | 解析 |
| :--- | :--- | :--- |
| **18. 梯度消失解決** | (A) 使用 ReLU 激活函數 | Sigmoid 在深層網路容易導致梯度消失，ReLU 可緩解此問題。 |
| **26. Batch Normalization** | (B) 加速收斂並抑制過擬合 | BN 層將數據標準化，讓權重更新更穩定。 |
| **45. Learning Rate 調整** | (C) 隨訓練次數衰減 (Decay) | 訓練初期用大 LR 快速收斂，後期用小 LR 微調。 |

## 3. 考試技巧 (Tips)
1.  **關鍵字聯想**：看到 "Time Series" 找 "RNN/LSTM/ARIMA"；看到 "Image" 找 "CNN"；看到 "Text" 找 "Transformer"。
2.  **刪去法**：IPAS 題目通常有兩個選項明顯錯誤，先刪掉它們，勝率直接提升到 50%。
3.  **注意「非」**：題目常問「下列何者**非**」，看錯就送分了。

## 4. 下一關預告
複習完考題後，我們要進入最後的進階主題。
Day 42 我們將探討 **MLOps (Machine Learning Operations)**，學習如何將模型部署到大規模生產環境 (Kubernetes)。
