# Day07_NaiveBayes_Complete.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests, zipfile, io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 設定繪圖風格
sns.set(style="whitegrid")

# ---------------------------------------------------------
# 1. 資料準備 (自動下載 UCI SMS Spam Collection)
# ---------------------------------------------------------
def load_uci_data():
    """從 UCI 下載並回傳 DataFrame，若失敗則回傳 None"""
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    print("1. 正在下載 UCI 資料集...")
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
if df is None: # 備用方案
    df = pd.DataFrame({'text': ['Free money', 'Hi mom'], 'label': ['spam', 'ham']})

# 將標籤轉為數字: spam=1, ham=0
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
print(f"   資料集準備完成，共 {len(df)} 筆資料。")

# ==============================================================================
# 2. 【教學插曲】詞袋模型視覺化演示 (Bag-of-Words Demo)
# ==============================================================================
print("-" * 30)
print("2. 啟動【教學插曲】：詞袋模型視覺化 (Bag-of-Words Demo)")
print("   讓我們用一個簡單的例子，看看電腦是怎麼「讀」文字的...")

# 準備演示用的微型資料
demo_corpus = [
    "Win cash now",  # 簡訊 A
    "Call now"       # 簡訊 B
]

# 初始化演示用的 Vectorizer
# token_pattern 設定是為了讓 'a', 'I' 這種單字也能被捕捉 (預設會忽略長度<2的字)
demo_vectorizer = CountVectorizer(stop_words=None, token_pattern=r"(?u)\b\w+\b")
X_demo = demo_vectorizer.fit_transform(demo_corpus)

# 轉成 DataFrame
demo_df = pd.DataFrame(X_demo.toarray(), 
                       columns=demo_vectorizer.get_feature_names_out(),
                       index=['Message A', 'Message B'])

# --- 繪製圖表 1：詞袋概念圖 ---
plt.figure(figsize=(8, 4))
sns.heatmap(demo_df, annot=True, cmap="Blues", cbar=False, 
            linewidths=1, linecolor='black', fmt='d', annot_kws={"size": 16})
plt.title("Figure 1: Bag-of-Words Concept Demo\n(Visualizing 'Win cash now' vs 'Call now')", fontsize=14)
plt.yticks(rotation=0)
plt.tick_params(axis='both', which='major', labelsize=12, labelbottom=False, labeltop=True)
plt.tight_layout()
plt.show() 
print("   ★ 圖表 1 已顯示：請關閉圖表視窗以繼續執行主程式...")

# ---------------------------------------------------------
# 3. 真實特徵工程 (UCI Data Processing)
# ---------------------------------------------------------
print("-" * 30)
print("3. 回到主線：處理 5000+ 筆真實簡訊...")

# 針對英文資料的設定
vectorizer = CountVectorizer(stop_words='english', max_features=3000)
X_counts = vectorizer.fit_transform(df['text'])
y = df['label_num']

print(f"   矩陣形狀: {X_counts.shape} (文件數, 單字特徵數)")

# ---------------------------------------------------------
# 4. 模型訓練 (Model Training)
# ---------------------------------------------------------
print("4. 訓練貝氏分類器 (Naive Bayes)...")
X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.2, random_state=42)

clf_nb = MultinomialNB(alpha=1.0) # alpha=1.0 為拉普拉斯平滑預設值
clf_nb.fit(X_train, y_train)

y_pred = clf_nb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"   ★ 模型測試集準確率: {acc:.2%}")

# ---------------------------------------------------------
# 5. 視覺化成果 (Final Result Visualization)
# ---------------------------------------------------------
print("5. 繪製最終成果圖...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 子圖 A：混淆矩陣
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0], annot_kws={"size": 14})
axes[0].set_title(f'Figure 2A: Confusion Matrix (UCI Data)\nAccuracy: {acc:.2%}', fontsize=14)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticklabels(['Ham', 'Spam'])
axes[0].set_yticklabels(['Ham', 'Spam'])

# 子圖 B：關鍵詞分析
feature_names = vectorizer.get_feature_names_out()
spam_prob = clf_nb.feature_log_prob_[1, :]
ham_prob = clf_nb.feature_log_prob_[0, :]
spamminess_score = spam_prob - ham_prob

words_df = pd.DataFrame({'word': feature_names, 'spamminess': spamminess_score})
top_spam = words_df.sort_values(by='spamminess', ascending=False).head(10)
top_ham = words_df.sort_values(by='spamminess', ascending=True).head(10)
top_words = pd.concat([top_ham, top_spam])

colors = ['#1f77b4' if x < 0 else '#d62728' for x in top_words['spamminess']]
axes[1].barh(top_words['word'], top_words['spamminess'], color=colors)
axes[1].set_title('Figure 2B: Top Keywords (Spam vs Ham)', fontsize=14)
axes[1].set_xlabel('Log Probability Difference (Spam - Ham)')
axes[1].axvline(0, color='black', linestyle='--')

plt.tight_layout()
plt.show()
print("★ 程式執行完畢！")