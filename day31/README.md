# Day 31: NLP 王者 - Transformer 與 BERT (The Age of LLM)

## 0. 前言：NLP 的工業革命
在 Day 27/28 我們學過了 RNN 和 LSTM。雖然它們能處理序列資料，但有兩個致命缺點，這也是 IPAS 考題中常問的「RNN 限制」：
1.  **無法平行運算 (Slow)**：必須等 $t-1$ 算完才能算 $t$，訓練速度慢，無法利用 GPU 的平行優勢。
2.  **長距離遺忘 (Long-term Dependency)**：雖然 LSTM 有改善，但當句子太長 (例如 1000 字) 時，開頭的資訊還是會丟失。

2017 年 Google 發表的論文 **"Attention Is All You Need"** 改變了一切。
**Transformer** 架構完全捨棄了迴圈 (Recurrence)，改用 **Self-Attention (自注意力機制)**。
這就像是從「傳話遊戲 (RNN)」進化到了「視訊會議 (Transformer)」，所有人可以直接跟所有人溝通。

## 1. 核心概念：Self-Attention (自注意力機制)
這是 IPAS 考試的**超級必考題**。你必須理解 Q, K, V 的物理意義。

### 1.1 直觀理解
想像你在讀這句話：「**蘋果**因為**它**很好吃所以被我吃了。」
*   當我們讀到「**它**」的時候，我們的大腦會自動把它連結到「**蘋果**」。
*   這就是 Attention！機器在處理每個字時，會去「關注 (Attend to)」句子中其他相關的字，計算它們之間的關聯強度。

### 1.2 Q, K, V 的比喻 (圖書館搜尋)
Transformer 把每個字都變成了三個向量：**Query (Q)**, **Key (K)**, **Value (V)**。

*   **Query (查詢)**：拿著這個字去「找」相關的資訊。(例如：我想找「它」指代什麼？)
*   **Key (索引)**：每個字身上的「標籤」。(例如：「蘋果」的標籤是「食物/水果」)
*   **Value (內容)**：這個字實際包含的「語意資訊」。(例如：「蘋果」的向量表示)

**運作流程**：
1.  **匹配 (Dot Product)**：拿目前的 $Q$ 去跟所有字的 $K$ 做內積。內積越大，代表關聯越強 (Attention Score 高)。
2.  **正規化 (Softmax)**：將分數轉化為機率 (總和為 1)。
3.  **加權總合 (Weighted Sum)**：根據機率，把所有相關字的 $V$ 加起來。

> **IPAS 考點公式**：
> $$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V $$
> *   為什麼要除以 $\sqrt{d_k}$？ **為了避免內積數值過大，導致 Softmax 進入梯度消失區間 (Gradient Vanishing)。**

## 2. Transformer 的兩大護法：BERT vs GPT
Transformer 架構後來分裂成了兩大派系，這也是面試與考試的熱門題。

| 特性 | **BERT** (Bidirectional Encoder Representations from Transformers) | **GPT** (Generative Pre-trained Transformer) |
| :--- | :--- | :--- |
| **架構** | **Encoder (編碼器)** | **Decoder (解碼器)** |
| **方向** | **雙向 (Bidirectional)** <br> 同時看上下文 (左到右 + 右到左)。 | **單向 (Unidirectional)** <br> 只能看前面 (左到右)，預測下一個字。 |
| **訓練任務** | **克漏字 (Masked LM)** <br> "今天天氣 [MASK] 好" -> 猜 "真" | **文字接龍 (Next Token Prediction)** <br> "今天天氣" -> 猜 "真" |
| **強項** | **理解 (Understanding)** <br> 文本分類、情緒分析、問答、實體辨識。 | **生成 (Generation)** <br> 寫文章、聊天、寫程式、創意發想。 |
| **代表模型** | BERT, RoBERTa, DistilBERT | GPT-3, GPT-4, Claude, LLaMA |

## 3. 實戰：使用 Hugging Face Transformers
我們不需要從頭刻 Transformer (那太痛苦了)。
**Hugging Face** 是 AI 界的 GitHub，提供了數十萬個預訓練模型與 `transformers` 套件。

### 3.1 安裝
```bash
pip install transformers torch
```

### 3.2 程式碼實作：情緒分析
完整程式連結：[Transformer_Sentiment.py](Transformer_Sentiment.py)

我們使用 `pipeline` API，這是最簡單的高階介面。它會自動幫我們做三件事：
1.  **Tokenization**：把文字切成 Token (數字)。
2.  **Model Inference**：丟進 BERT 模型運算。
3.  **Post-processing**：把模型輸出的 Logits 轉成標籤 (Positive/Negative)。

```python
from transformers import pipeline

# 強制使用 PyTorch (framework="pt")
classifier = pipeline("sentiment-analysis", framework="pt")

result = classifier("IPAS certification is challenging but worth it.")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

## 4. IPAS 考前重點複習
1.  **Transformer 優於 RNN 的主因？**
    *   可以**平行運算** (Parallelization)，訓練速度快。
    *   解決了**長距離依賴** (Long-term Dependency) 問題。
2.  **Self-Attention 的計算複雜度？**
    *   與序列長度 $N$ 的平方成正比 ($O(N^2)$)。所以 Transformer 很難處理超長文本 (如整本書)。
3.  **BERT 的預訓練任務？**
    *   Masked Language Model (MLM, 克漏字)。
    *   Next Sentence Prediction (NSP, 預測下一句)。
4.  **什麼是 Fine-tuning (微調)？**
    *   拿 Google 訓練好的通用 BERT (預訓練模型)，接上自己的資料 (例如公司客服紀錄)，稍微訓練一下，讓它變成專用模型。這就是 **遷移學習 (Transfer Learning)** 的極致應用。

## 5. 下一關預告
Day 32 我們將進入電腦視覺的進階領域：**物件偵測 (Object Detection)**。
我們要從「這張圖是什麼 (Classification)」進化到「這東西在哪裡 (Detection)」。
主角是速度極快的 **YOLO (You Only Look Once)**！
