# Day 38: 生成式 AI 與 LLM 應用 (GenAI & LLM Applications)

## 1. 前言：從「預測」到「創造」
前 37 天我們學的 (Regression, Classification, RL) 大多是 **判別式 AI (Discriminative AI)**：
*   **任務**：分辨這是貓還是狗？預測房價是多少？
*   **本質**：尋找數據之間的邊界或規律。

今天我們要進入 **生成式 AI (Generative AI)** 的世界：
*   **任務**：畫一隻貓、寫一首詩、寫程式碼。
*   **本質**：學習數據的分布，並創造出全新的數據。

## 2. LLM (大型語言模型) 崛起
LLM (Large Language Model) 是 GenAI 的核心，代表作如 GPT-4, Gemini, Claude, LLaMA。
它們的基礎是我們在 Day 31 學過的 **Transformer** 架構 (主要是 Decoder-Only)。

### 2.1 兩大應用流派
要讓 LLM 為我們工作，主要有兩種方式：

| 方法 | 說明 | 比喻 | 成本 |
| :--- | :--- | :--- | :--- |
| **RAG (檢索增強生成)** | 給 AI 一本教科書 (外部資料)，讓它翻書回答。 | **開卷考試** | 低 (只需維護資料庫) |
| **Fine-tuning (微調)** | 拿特定領域的資料重新訓練 AI，改變它的腦袋。 | **送去補習班** | 高 (需要算力訓練) |

## 3. 核心技術：RAG (Retrieval-Augmented Generation)
為什麼需要 RAG？因為 LLM 有兩個大毛病：
1.  **幻覺 (Hallucination)**：一本正經地胡說八道。
2.  **知識過期**：它只知道訓練截止日之前的事。

**RAG 的運作流程**：
1.  **User 提問**：「公司最新的請假規定是什麼？」
2.  **Retrieval (檢索)**：系統去公司的文檔庫搜尋相關文章。
3.  **Augmentation (增強)**：把「找到的文章」+「使用者的問題」組合成一個 Prompt。
    *   *Prompt: "請根據以下文章回答問題：[文章內容]... 問題：公司最新的請假規定是什麼？"*
4.  **Generation (生成)**：LLM 根據 Prompt 生成正確答案。

## 4. 實戰：本地端 LLM 體驗 (Hugging Face)
我們使用 Python 的 `transformers` 庫來體驗最基礎的文字生成。
為了在個人電腦上跑得動，我們使用輕量級的 **GPT-2** 模型。

### 4.1 程式碼架構 (`LLM_Demo.py`)
1.  **Pipeline**：Hugging Face 最方便的工具，一行程式碼載入模型。
2.  **Text Generation**：輸入開頭 (Prompt)，讓 AI 接龍。
3.  **Prompt Engineering**：展示如何透過「給範例 (Few-Shot)」讓 AI 表現得更好。

## 5. 執行結果預期
*   **Zero-Shot (直接問)**：AI 可能會亂接龍，寫出不通順的句子。
*   **Few-Shot (給範例)**：AI 模仿範例的格式，產出比較合理的結果。

## 6. 下一關預告
AI 越來越強，但也越來越像「黑盒子」。
Day 39 我們將探討 **XAI (可解釋 AI)** 與 **AI 治理**。
為什麼 AI 會這樣判斷？我們該如何監管它？
這是邁向 AI 資深工程師的必修課。
