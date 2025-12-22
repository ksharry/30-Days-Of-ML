import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# NLP 工具
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score

# --- 1. 載入資料 (Data Loading) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'spam.csv')

if not os.path.exists(DATA_FILE):
    print(f"錯誤：找不到檔案 {DATA_FILE}")
    print("請手動下載資料集或檢查路徑。")
    exit()

# 這個 csv 通常是 latin-1 編碼
try:
    df = pd.read_csv(DATA_FILE, encoding='latin-1')
except:
    df = pd.read_csv(DATA_FILE, encoding='utf-8')

# --- 2. 資料清理與 EDA ---
# 原始資料有很多無用的欄位 (Unnamed: 2, 3, 4)，且欄位名稱不直觀
df = df[['v1', 'v2']]
df.columns = ['Category', 'Message']

print(f"資料集維度: {df.shape}")
print(df.head())

# 檢查類別分佈
print(df['Category'].value_counts())

pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

plt.figure(figsize=(6, 4))
sns.countplot(x='Category', data=df, palette='viridis')
plt.title('Spam vs Ham Distribution')
plt.savefig(os.path.join(pic_dir, '8-1_Class_Distribution.png'))

# 將類別轉為數字 (Ham=0, Spam=1)
df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# --- 3. 文字前處理 (NLP Basics) ---
# 我們使用 Bag of Words (詞袋模型)
# CountVectorizer 會幫我們做幾件事：
# 1. Tokenization (斷詞)
# 2. Lowercase (轉小寫)
# 3. Stopwords removal (去除停用詞，如 the, is, a) - 這裡我們先用預設英文停用詞
vectorizer = CountVectorizer(stop_words='english')

X = vectorizer.fit_transform(df['Message'])
y = df['Spam']

print(f"詞彙表大小 (Vocabulary Size): {len(vectorizer.get_feature_names_out())}")
# print(vectorizer.get_feature_names_out()[1000:1010]) # 看看抓到了什麼詞

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# --- 4. 建立與訓練模型 ---
# 對於文字計數資料 (Discrete Counts)，MultinomialNB 是最適合的
model = MultinomialNB()
model.fit(X_train, y_train)

# --- 5. 模型評估 ---
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics_output = f"""
Accuracy: {acc:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1:.4f}
Confusion Matrix:
{cm}

Classification Report:
{classification_report(y_test, y_pred)}
"""

print(metrics_output)
with open(os.path.join(SCRIPT_DIR, 'metrics.txt'), 'w') as f:
    f.write(metrics_output)

# 繪製混淆矩陣
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix (Naive Bayes)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(pic_dir, '8-2_Confusion_Matrix.png'))

# --- 額外：看看模型學到了什麼 (Top Spam Words) ---
# 取得每個詞在 Spam 類別的機率 (Log Probability)
feature_names = vectorizer.get_feature_names_out()
spam_prob_sorted = model.feature_log_prob_[1].argsort()[::-1] # 1 是 Spam 類別

top_10_spam_words = np.take(feature_names, spam_prob_sorted[:10])
print(f"Top 10 Spam Words: {top_10_spam_words}")

# 畫個簡單的長條圖顯示 Top 10 Spam Words
plt.figure(figsize=(10, 6))
# 為了畫圖，我們取回對應的 log prob (雖然是負數，但相對大小可見)
top_10_probs = np.take(model.feature_log_prob_[1], spam_prob_sorted[:10])
sns.barplot(x=top_10_probs, y=top_10_spam_words, palette='Reds_r')
plt.title('Top 10 Keywords in Spam Messages (Log Probability)')
plt.xlabel('Log Probability (Higher is more likely)')
plt.savefig(os.path.join(pic_dir, '8-3_Top_Spam_Words.png'))
