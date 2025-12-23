# Day 22: NLP Bag of Words (BoW) - 垃圾郵件分類
# ---------------------------------------------------------
# 這一天的目標是進入自然語言處理 (NLP) 的世界。
# 我們要學習最基礎的文字表示法：Bag of Words (詞袋模型)。
# 它的概念是：不看文法、不看順序，只看「關鍵字出現了幾次」。
# 我們將使用 SMS Spam Collection 資料集來實作垃圾簡訊分類。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import urllib.request
import zipfile

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 下載 SMS Spam Collection 資料集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
zip_path = os.path.join(SCRIPT_DIR, "smsspamcollection.zip")
data_path = os.path.join(SCRIPT_DIR, "SMSSpamCollection")

if not os.path.exists(data_path):
    print("Downloading SMS Spam Collection Dataset...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(SCRIPT_DIR)
    print("Download and extraction complete.")

# 讀取資料
# 格式：label (tab) message
df = pd.read_csv(data_path, sep='\t', names=['label', 'message'])

print("Data loaded:")
print(f"Shape: {df.shape}")
print(df.head())

# 轉換標籤：ham (正常) -> 0, spam (垃圾) -> 1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# --- 2. 文字特徵提取 (Feature Extraction) - Bag of Words ---
# CountVectorizer 會幫我們做三件事：
# 1. Tokenization (斷詞)：把句子切成單字。
# 2. Vocabulary Building (建立字典)：找出所有出現過的單字。
# 3. Encoding (編碼)：計算每個單字在每句話出現的次數。

vectorizer = CountVectorizer(stop_words='english') # 去除停用詞 (the, is, at...)
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

print(f"\nVocabulary size: {len(vectorizer.get_feature_names_out())}")
print(f"Feature matrix shape: {X.shape}")

# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. 訓練模型 (Naive Bayes) ---
# Naive Bayes (樸素貝氏) 是處理文字分類的經典演算法
# MultinomialNB 專門處理「次數」類型的特徵 (如 BoW)
model = MultinomialNB()
model.fit(X_train, y_train)

# --- 4. 模型評估 (Evaluation) ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 繪製混淆矩陣
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)

# 印出數值供 README 使用
tn, fp, fn, tp = cm.ravel()
print(f"\n--- Confusion Matrix Values ---")
print(f"TN (True Negative - 正常簡訊): {tn}")
print(f"FP (False Positive - 誤判為垃圾): {fp}")
print(f"FN (False Negative - 漏抓垃圾): {fn}")
print(f"TP (True Positive - 抓到垃圾): {tp}")
print(f"Total: {tn + fp + fn + tp}")
print("-------------------------------\n")

sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix (Naive Bayes)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(pic_dir, '22-1_Confusion_Matrix.png'))
print("Confusion Matrix saved.")

# --- 5. 視覺化：最常出現的垃圾關鍵字 (Top Spam Words) ---
# 找出哪些字最常出現在垃圾簡訊中
# 1. 取得所有單字
feature_names = vectorizer.get_feature_names_out()
# 2. 取得 Spam 類的特徵總和
spam_index = 1
spam_words_count = model.feature_count_[spam_index]
# 3. 排序並取前 20 名
top_indices = spam_words_count.argsort()[-20:][::-1]
top_words = [feature_names[i] for i in top_indices]
top_counts = [spam_words_count[i] for i in top_indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=top_counts, y=top_words, palette='Reds_r')
plt.title('Top 20 Words in Spam Messages')
plt.xlabel('Frequency')
plt.savefig(os.path.join(pic_dir, '22-2_Top_Spam_Words.png'))
print("Top Spam Words plot saved.")
