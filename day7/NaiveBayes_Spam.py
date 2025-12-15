# Day07_NaiveBayes_UCI.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests, zipfile, io # 用於下載資料
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------
# 1. 資料準備 (自動下載 UCI SMS Spam Collection)
# ---------------------------------------------------------
def load_uci_data():
    """從 UCI 下載並回傳 DataFrame，若失敗則回傳 None"""
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    print("正在下載 UCI 資料集...")
    try:
        r = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open('SMSSpamCollection') as f:
                df = pd.read_csv(f, sep='\t', header=None, names=['label', 'text'])
        return df
    except Exception as e:
        print(f"下載失敗: {e}")
        return None

# 讀取資料
df = load_uci_data()

# 如果下載失敗，為了讓程式能跑，我們手動建立一個極簡備用資料
if df is None:
    data = {'text': ['Free money', 'Hi mom'], 'label': ['spam', 'ham']}
    df = pd.DataFrame(data)

# 資料前處理
# 1. 將標籤轉為數字: spam=1, ham=0
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
X_raw = df['text']
y = df['label_num']

print(f"資料集準備完成，共 {len(df)} 筆資料。")
print(f"垃圾信 (Spam): {len(df[df['label']=='spam'])} 筆")
print(f"正常信 (Ham):  {len(df[df['label']=='ham'])} 筆")
print("-" * 30)

# ---------------------------------------------------------
# 2. 特徵工程：文字轉數字 (Bag of Words)
# ---------------------------------------------------------
# 針對英文資料的設定：
# stop_words='english': 去除 'the', 'is', 'at' 這種無意義的高頻詞
# max_features=3000: 只取最常出現的前 3000 個字，避免特徵太多運算太慢
vectorizer = CountVectorizer(stop_words='english', max_features=3000)

print("正在進行文字向量化 (這可能需要一點時間)...")
X_counts = vectorizer.fit_transform(X_raw)

print(f"矩陣形狀: {X_counts.shape} (文件數, 單字特徵數)")

# ---------------------------------------------------------
# 3. 模型訓練 (Model Training)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.2, random_state=42)

clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)

y_pred = clf_nb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"★ 模型測試集準確率: {acc:.2%}")

# ---------------------------------------------------------
# 4. 視覺化成果 (Visualization)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 圖一：混淆矩陣
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0], annot_kws={"size": 14})
axes[0].set_title(f'UCI SMS Spam - Naive Bayes\nAccuracy: {acc:.2%}', fontsize=14)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticklabels(['Ham', 'Spam'])
axes[0].set_yticklabels(['Ham', 'Spam'])

# 圖二：關鍵詞分析 (Top Spammy vs Hammy Words)
feature_names = vectorizer.get_feature_names_out()
spam_prob = clf_nb.feature_log_prob_[1, :]
ham_prob = clf_nb.feature_log_prob_[0, :]

# 計算垃圾度 (Spamminess)
spamminess_score = spam_prob - ham_prob

words_df = pd.DataFrame({
    'word': feature_names,
    'spamminess': spamminess_score
})

# 取出最垃圾的 10 個詞和最正常的 10 個詞
top_spam = words_df.sort_values(by='spamminess', ascending=False).head(10)
top_ham = words_df.sort_values(by='spamminess', ascending=True).head(10)
top_words = pd.concat([top_ham, top_spam])

colors = ['#1f77b4' if x < 0 else '#d62728' for x in top_words['spamminess']]
axes[1].barh(top_words['word'], top_words['spamminess'], color=colors)
axes[1].set_title('Top English Keywords for Spam (Red) vs Ham (Blue)', fontsize=14)
axes[1].set_xlabel('Log Probability Difference (Spam - Ham)')
axes[1].axvline(0, color='black', linestyle='--')

plt.tight_layout()
plt.show()

# 印出關鍵字給你看
print("\n【分析結果】")
print("最常出現在 UCI 垃圾簡訊的關鍵字 (Top Spam Words):")
print(top_spam['word'].values)