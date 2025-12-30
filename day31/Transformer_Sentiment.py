import os
# 設定環境變數以隱藏 TensorFlow 的警告訊息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformers import pipeline

def main():
    print("=== Day 31: Transformer Sentiment Analysis ===")
    print("正在載入預訓練模型 (DistilBERT)... 請稍候")
    
    # 1. 建立 Pipeline
    #這會自動下載一個微調過的 BERT 模型 (預設是 distilbert-base-uncased-finetuned-sst-2-english)
    # pipeline 是 Hugging Face 最簡單的高階 API，直接封裝了 Tokenization -> Model -> Post-processing
    # 強制使用 PyTorch (framework="pt") 以避免 TensorFlow 版本衝突
    classifier = pipeline("sentiment-analysis", framework="pt")
    
    # 2. 準備測試資料
    test_sentences = [
        "I love learning machine learning, it is so fascinating!",  # 正面
        "I am very disappointed with the service, it was terrible.", # 負面
        "The food was okay, but the atmosphere was a bit noisy.",    # 混合/負面?
        "IPAS certification is challenging but worth it."            # 正面
    ]
    
    # 3. 進行預測
    print("\n=== 預測結果 ===")
    results = classifier(test_sentences)
    
    for text, res in zip(test_sentences, results):
        label = res['label']
        score = res['score']
        
        # 簡單的視覺化
        sentiment_icon = "😊" if label == "POSITIVE" else "😞"
        
        print(f"句子: {text}")
        print(f"情緒: {sentiment_icon} {label} (信心度: {score:.4f})")
        print("-" * 50)

if __name__ == "__main__":
    main()
