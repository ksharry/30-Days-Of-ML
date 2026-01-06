import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, set_seed

# 設定隨機種子
set_seed(42)

# === 1. 準備私有知識庫 (Knowledge Base) ===
# 假設這是我們公司內部的文檔，ChatGPT 訓練資料裡絕對沒有
# 我們將資料改為英文，以配合 GPT-2 的生成能力
documents = [
    "30-Days-Of-ML is a machine learning challenge initiated by Harry.",
    "In Day 37, we learned the DQN algorithm to play the CartPole game.",
    "The topic of Day 39 was XAI (Explainable AI), using the SHAP library.",
    "Harry's cat is named 'Oreo', and it loves sleeping on the keyboard.",
    "The ultimate goal of this project is to build a RAG system."
]

print(f"知識庫載入完成，共有 {len(documents)} 筆資料。")

# === 2. 向量化 (Embedding) ===
# 使用標準的英文 Embedding 模型，效果比多語言版本更精準
print("正在載入 Embedding 模型 (sentence-transformers)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

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
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    
    print("-" * 30)
    print("【AI 生成中...】")
    # 設定 max_new_tokens 避免生成過短或過長
    output = generator(prompt, max_new_tokens=50, num_return_sequences=1, truncation=True)
    
    # 只顯示生成的回答部分 (去掉 Prompt)
    generated_text = output[0]['generated_text']
    print(generated_text.replace(prompt, "").strip())

# === 5. 測試 ===
# 測試 1: 問專案內容 (英文提問)
rag_pipeline("What is the name of Harry's cat?")

# 測試 2: 問技術細節 (英文提問)
rag_pipeline("What did we learn in Day 39?")
