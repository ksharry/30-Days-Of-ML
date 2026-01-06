from transformers import pipeline, set_seed

# 設定隨機種子，讓結果可重現
set_seed(42)

def run_llm_demo():
    print("正在載入 GPT-2 模型 (這可能需要一點時間下載)...")
    # 使用 Hugging Face 的 pipeline，任務是 "text-generation"
    # model='gpt2' 是一個比較小的模型，適合 CPU 執行
    generator = pipeline('text-generation', model='gpt2')
    
    print("\n=== 1. 基礎生成 (Zero-Shot) ===")
    prompt = "Artificial Intelligence is"
    print(f"Prompt: {prompt}")
    
    # max_length: 生成的最大長度
    # num_return_sequences: 生成幾個版本
    # truncation=True: 避免過長
    result = generator(prompt, max_length=50, num_return_sequences=1, truncation=True)
    print(f"Generated:\n{result[0]['generated_text']}")
    print("-" * 30)

    print("\n=== 2. 提示工程 (Few-Shot Prompting) ===")
    # 透過給予範例，引導模型做特定的事 (例如：英翻中，雖然 GPT-2 中文能力很差，這裡用情感分析分類做示範)
    # 格式：[Input] -> [Label]
    few_shot_prompt = """
    Review: This movie is amazing! -> Positive
    Review: The food was terrible. -> Negative
    Review: I love this product. -> Positive
    Review: The service was bad. ->"""
    
    print(f"Prompt (給範例):\n{few_shot_prompt}")
    
    result = generator(few_shot_prompt, max_length=60, num_return_sequences=1, truncation=True)
    
    # 只印出生成的部份
    generated_text = result[0]['generated_text']
    # 簡單處理一下輸出，只看最後一行
    print(f"Generated Result: {generated_text.split('->')[-1].strip()}")
    print("-" * 30)
    
    print("\n=== 3. 創意寫作 (Creative Writing) ===")
    story_prompt = "Once upon a time, a robot learned to love,"
    print(f"Prompt: {story_prompt}")
    
    # temperature: 創意度 (越高越瘋狂，越低越保守)
    # top_k: 每次只從機率最高的 k 個字選
    result = generator(story_prompt, max_length=100, temperature=0.9, top_k=50, num_return_sequences=1, truncation=True)
    print(f"Generated Story:\n{result[0]['generated_text']}")

if __name__ == "__main__":
    # 檢查是否有安裝 transformers
    try:
        import transformers
        run_llm_demo()
    except ImportError:
        print("請先安裝 transformers: pip install transformers torch")
