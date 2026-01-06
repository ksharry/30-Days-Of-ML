import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, set_seed

# 設定隨機種子
set_seed(42)

# === 1. 準備私有知識庫 (Knowledge Base) ===
# 假設這是我們公司內部的文檔，ChatGPT 訓練資料裡絕對沒有
documents = [
    "30-Days-Of-ML 是一個由 Harry 發起的機器學習挑戰計畫。",
    "在 Day 37 中，我們學習了 DQN 演算法來玩 CartPole 遊戲。",
    "Day 39 的主題是 XAI (可解釋 AI)，使用了 SHAP 套件。",
    "Harry 的貓叫做 'Oreo'，牠喜歡睡在鍵盤上。",
    "這個專案的最終目標是建立一個 RAG 系統。"
]

print(f"知識庫載入完成，共有 {len(documents)} 筆資料。")

# === 2. 向量化 (Embedding) ===
# 載入一個輕量級的中文/多語言 Embedding 模型
# 'paraphrase-multilingual-MiniLM-L12-v2' 支援中文，且速度快
print("正在載入 Embedding 模型 (sentence-transformers)...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 把所有文檔變成向量
doc_embeddings = embedder.encode(documents)
print(f"文檔向量化完成。維度: {doc_embeddings.shape}")

# === 3. 檢索函式 (Retrieval) ===
def search(query, top_k=2):
    # 1. 把使用者的問題變成向量
    query_embedding = embedder.encode([query])
    
    # 2. 計算相似度 (Cosine Similarity)
    # 公式: (A . B) / (|A| * |B|)
    # 因為 sentence-transformers 輸出的向量已經正規化過，所以直接算內積 (dot product) 就好
    similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
    
    # 3. 找出分數最高的 top_k 個索引
    # argsort 是由小排到大，所以取最後 k 個並反轉
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_k_indices:
        results.append((documents[idx], similarities[idx]))
        
    return results

# === 4. 生成函式 (Generation) ===
print("正在載入生成模型 (GPT-2)...")
# 為了演示方便，我們用 GPT-2 (英文模型)，實際應用通常會接 OpenAI API 或 Llama 3
generator = pipeline('text-generation', model='gpt2')

def rag_pipeline(query):
    print(f"\n使用者問題: {query}")
    print("-" * 30)
    
    # Step 1: 檢索 (Retrieve)
    retrieved_docs = search(query)
    print("【檢索到的相關資料】:")
    context = ""
    for i, (doc, score) in enumerate(retrieved_docs):
        print(f"{i+1}. {doc} (相似度: {score:.4f})")
        context += doc + " "
    
    # Step 2: 生成 (Generate)
    # 組合 Prompt
    # 注意: 因為 GPT-2 是英文模型，我們這裡用英文 Prompt 格式演示結構
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    
    print("-" * 30)
    print("【AI 生成中...】")
    # 這裡生成的品質取決於模型大小，GPT-2 對中文支援極差，主要看它是否能參考 Context
    output = generator(prompt, max_length=150, num_return_sequences=1, truncation=True)
    print(output[0]['generated_text'])

# === 5. 測試 ===
# 測試 1: 問專案內容
rag_pipeline("Harry 的貓叫什麼名字？")

# 測試 2: 問技術細節
rag_pipeline("Day 39 教了什麼？")
