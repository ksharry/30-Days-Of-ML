# Day07_NaiveBayes_Spam.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------
# 1. 資料準備 (建立一個迷你垃圾郵件資料集)
# ---------------------------------------------------------
# 為了演示方便，我們手動建立一個小型的合成資料集。
# 在真實場景中，這裡會是讀取數千封郵件的 CSV 檔。
data = {
    'text': [
        '恭喜你中獎了！馬上點擊領取百萬獎金', # Spam
        '獨家優惠，限時搶購，這檔股票會飆',   # Spam
        '免費比特幣，現在註冊就送',           # Spam
        '你的帳戶存在風險，請立即驗證密碼',   # Spam
        '這種減肥藥效果驚人，一個月瘦十公斤', # Spam
        '今天晚上要不要一起去吃晚餐？',       # Ham
        '上次開會的會議記錄請查收',           # Ham
        '明天早上九點的專案進度報告',         # Ham
        '媽媽寄的包裹已經到了，記得去拿',     # Ham
        '這週六要不要去爬山？天氣好像不錯'    # Ham
    ],
    'label': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] # 1=Spam(垃圾信), 0=Ham(正常信)
}
df = pd.DataFrame(data)
X_raw = df['text']
y = df['label']

print("原始文字資料範例：")
print(X_raw.head(3))
print("-" * 30)

# ==============================================================================
# 【插曲】處理文字特徵：從文字到數字 (The Interlude: From Text to Numbers)
# ==============================================================================
# 電腦看不懂中文，我們必須把它變成數字矩陣。
# 我們使用「詞袋模型 (Bag-of-Words)」，簡單來說就是數人頭：計算每個詞出現的次數。

# 1. 初始化向量化器 (這裡簡單使用以字元為單位，實際應用會需要中文斷詞庫如 Jieba)
# analyzer='char_wb' 這裡為了簡化演示不依賴額外斷詞庫，將以字元(n-gram)切分
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))

# 2. 將文字轉換為「文件-詞項矩陣 (Document-Term Matrix)」
X_counts = vectorizer.fit_transform(X_raw)

# 3. 讓我們看看轉換後長什麼樣子 (Demo)
print("【插曲演示】文字向量化後的樣子 (前兩筆)：")
# 取得詞彙表 (Feature Names)
feature_names = vectorizer.get_feature_names_out()
# 轉成 DataFrame 方便閱讀
df_demo = pd.DataFrame(X_counts[:2].toarray(), columns=feature_names)
# 只顯示部分欄位
print(df_demo.iloc[:, :10])
print(f"\n整個矩陣形狀: {X_counts.shape} (10封信, 產生了 {X_counts.shape[1]} 個特徵詞組)")
print("=" * 50)

# ---------------------------------------------------------
# 2. 模型訓練 (Model Training)
# ---------------------------------------------------------
# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.3, random_state=42)

# 初始化貝氏分類器 (使用 MultinomialNB，適合計數資料)
# alpha 是平滑參數 (Laplace Smoothing)，防止某個詞沒出現過導致機率變 0
clf_nb = MultinomialNB(alpha=1.0)
clf_nb.fit(X_train, y_train)

# 預測
y_pred = clf_nb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"模型準確率: {acc:.2%}")

# ---------------------------------------------------------
# 3. 視覺化成果 (Visualization)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 圖一：混淆矩陣 (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0], annot_kws={"size": 14})
axes[0].set_title(f'Naive Bayes Confusion Matrix\nAccuracy: {acc:.2%}', fontsize=14)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticklabels(['Ham (正常)', 'Spam (垃圾)'])
axes[0].set_yticklabels(['Ham (正常)', 'Spam (垃圾)'])

# 圖二：揭開貝氏的腦袋 - 最「垃圾」與最「正常」的關鍵詞 (Feature Importance)
# 貝氏模型內部儲存了每個詞在不同類別下的「對數機率 (Log Probability)」
# feature_log_prob_ 的形狀是 (類別數, 特徵數)
spam_prob = clf_nb.feature_log_prob_[1, :] # 垃圾信類別的詞機率
ham_prob = clf_nb.feature_log_prob_[0, :]  # 正常信類別的詞機率

# 計算「垃圾度」差異 (Spamminess) = P(詞|Spam) - P(詞|Ham)
# 差異越大，代表這個詞越傾向出現在垃圾信中
spamminess_score = spam_prob - ham_prob

# 建立 DataFrame 來排序
words_df = pd.DataFrame({
    'word': feature_names,
    'spamminess': spamminess_score
})

# 取出最垃圾的 10 個詞和最正常的 10 個詞
top_spam = words_df.sort_values(by='spamminess', ascending=False).head(10)
top_ham = words_df.sort_values(by='spamminess', ascending=True).head(10)
top_words = pd.concat([top_ham, top_spam])

# 繪製水平長條圖
colors = ['#1f77b4' if x < 0 else '#d62728' for x in top_words['spamminess']]
axes[1].barh(top_words['word'], top_words['spamminess'], color=colors)
axes[1].set_title('Top "Hammy" (Blue) vs. "Spammy" (Red) Words\n(Based on Log-Prob Difference)', fontsize=14)
axes[1].set_xlabel('Spamminess Score (Higher = More likely Spam)')
axes[1].axvline(0, color='black', linestyle='--')

plt.tight_layout()
# 由於有中文字，需要設定字型，若執行有問題可先註解掉或換成英文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.show()

print("\n【分析】")
print("貝氏模型學到了什麼？")
print("最常在垃圾信出現的特徵詞組 (Top Spammy):")
print(top_spam['word'].values)