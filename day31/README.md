# Day 31: Transformer 與 BERT (The Age of LLM)

## 1. 前言：NLP 的工業革命
在 Day 27/28 我們學過了 RNN 和 LSTM。雖然它們能處理序列資料，但有兩個致命缺點：
1.  **無法平行運算 (Slow)**：必須等 t-1 算完才能算 t，訓練速度慢。
2.  **長距離遺忘 (Long-term Dependency)**：雖然 LSTM 有改善，但距離太遠還是會忘記。

2017 年 Google 發表的 **Transformer** 架構解決了這一切。它完全捨棄了迴圈 (Recurrence)，改用 **Self-Attention (自注意力機制)**。
這就像是從「手寫信 (RNN)」進化到了「網際網路 (Transformer)」。

## 2. 核心概念：Self-Attention (自注意力機制)
這是 IPAS 考試的**超級必考題**。
想像你在讀這句話：「**蘋果**因為**它**很好吃所以被我吃了。」
*   當我們讀到「**它**」的時候，我們的大腦會自動把它連結到「**蘋果**」。
*   這就是 Attention！機器在處理每個字時，會去「關注」句子中其他相關的字。

### 2.1 Q, K, V 的比喻 (圖書館搜尋)
Transformer 把每個字都變成了三個向量：**Query (Q)**, **Key (K)**, **Value (V)**。
*   **Query (查詢)**：你想找什麼？ (例如：我想找「它」指代什麼？)
*   **Key (索引)**：圖書館裡的分類標籤。 (例如：「蘋果」的標籤是「食物/水果」)
*   **Value (內容)**：書本裡的實際內容。 (例如：「蘋果」的語意向量)

**運作流程**：
1.  拿著 Q 去跟所有的 K 做匹配 (點積運算 Dot Product)。
2.  如果 Q 和 K 很像 (匹配度高)，就取出對應的 V。
3.  最後把這些 V 加權總合，就得到了這個字在當下語境的意思。

> **公式 (必背)**：
> $$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V $$

## 3. BERT vs GPT
Transformer 架構後來分裂成了兩大派系：

| 特性 | **BERT** (Bidirectional Encoder Representations from Transformers) | **GPT** (Generative Pre-trained Transformer) |
| :--- | :--- | :--- |
| **架構** | **Encoder (編碼器)** | **Decoder (解碼器)** |
| **方向** | **雙向 (Bidirectional)** <br> 同時看上下文，理解能力強。 | **單向 (Unidirectional)** <br> 只能看前面，預測下一個字。 |
| **強項** | **理解 (Understanding)** <br> 分類、問答、實體辨識。 | **生成 (Generation)** <br> 寫文章、聊天、寫程式。 |
| **代表** | Google Search, IPAS 考題 | ChatGPT, Claude |

## 4. 實戰：Hugging Face Transformers
我們今天使用 Hugging Face 的 `pipeline` 來體驗 BERT 的威力。
Hugging Face 是 AI 界的 GitHub，提供了數十萬個預訓練模型。

### 程式碼解析 (`Transformer_Sentiment.py`)
```python
from transformers import pipeline

# 1. 下載模型
# pipeline 會自動下載一個微調過的 DistilBERT 模型
classifier = pipeline("sentiment-analysis")

# 2. 預測
result = classifier("I love machine learning!")
# 輸出: [{'label': 'POSITIVE', 'score': 0.9998}]
```
*   **DistilBERT**：是 BERT 的輕量版 (蒸餾版)，保留了 97% 的性能，但速度快 40%。

## 5. IPAS 考點複習
1.  **Transformer 解決了什麼問題？** 解決了 RNN 無法平行運算的問題。
2.  **Self-Attention 的三個核心向量？** Query, Key, Value。
3.  **BERT 是 Encoder 還是 Decoder？** Encoder (雙向)。
4.  **GPT 是 Encoder 還是 Decoder？** Decoder (單向)。

## 6. 下一關預告
Day 32 我們將進入電腦視覺的進階領域：**物件偵測 (Object Detection)**。
我們要從「這張圖是什麼 (Classification)」進化到「這東西在哪裡 (Detection)」。
主角是速度極快的 **YOLO (You Only Look Once)**！
