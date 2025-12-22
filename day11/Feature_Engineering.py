# Day 11: 特徵工程 (Feature Engineering) - 綜合案例
# ---------------------------------------------------------
# 這一天的目標是學習如何讓模型變強的關鍵：特徵工程。
# 我們使用一個不平衡的信用卡詐欺資料集 (模擬) 來展示 SMOTE 和特徵選擇的效果。
# 重點：特徵選擇 (RFE), 處理類別不平衡 (SMOTE)。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 模型與工具
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_selection import RFE
# --- 1. 產生模擬資料 (Data Generation) ---
# 我們產生一個高度不平衡的資料集，模擬信用卡詐欺偵測
# 1000 筆資料，20 個特徵，但只有 10% 是詐欺 (Class 1)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=2,
                           n_classes=2, weights=[0.9, 0.1], random_state=42)

feature_names = [f"Feature_{i}" for i in range(20)]
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

print(f"原始資料集維度: {df.shape}")
print(f"類別分佈:\n{df['Target'].value_counts()}")

# 視覺化原始類別分佈
plt.figure(figsize=(6, 4))
sns.countplot(x='Target', data=df, palette='pastel')
plt.title('Original Class Distribution (Imbalanced)')
plt.savefig(os.path.join(pic_dir, '11-1_Original_Distribution.png'))

# --- 2. 基準模型 (Baseline Model) ---
# 先不做任何處理，看看效果如何
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf_base = RandomForestClassifier(random_state=42)
clf_base.fit(X_train, y_train)
y_pred_base = clf_base.predict(X_test)

print("\n--- Baseline Model Results ---")
print(classification_report(y_test, y_pred_base))

# --- 3. 特徵選擇 (Feature Selection) - RFE ---
# 假設我們想從 20 個特徵中挑出最重要的 5 個
# RFE (Recursive Feature Elimination) 會遞迴地刪除最不重要的特徵
selector = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5, step=1)
selector = selector.fit(X_train, y_train)

selected_features = np.array(feature_names)[selector.support_]
print(f"\nRFE 挑選出的 Top 5 特徵: {selected_features}")

# 視覺化特徵排名
ranking = selector.ranking_
plt.figure(figsize=(10, 6))
plt.bar(range(len(ranking)), ranking, color='skyblue')
plt.xticks(range(len(ranking)), feature_names, rotation=90)
plt.ylabel('Rank (1 is best)')
plt.title('Feature Ranking by RFE (Lower is Better)')
plt.axhline(y=1.5, color='r', linestyle='--', label='Selected Features')
plt.legend()
plt.savefig(os.path.join(pic_dir, '11-2_RFE_Ranking.png'))

# --- 4. 處理類別不平衡 (Handling Imbalance) - Manual Oversampling ---
# 因為 Python 3.7 環境下 imblearn 可能有相容性問題，我們手動實作簡單的 Oversampling
# 也就是隨機複製少數類別的樣本

print("\n--- Applying Manual Oversampling ---")

# 分離多數與少數類別
X_train_df = pd.DataFrame(X_train, columns=feature_names)
y_train_df = pd.Series(y_train, name='Target')
train_data = pd.concat([X_train_df, y_train_df], axis=1)

majority_class = train_data[train_data['Target'] == 0]
minority_class = train_data[train_data['Target'] == 1]

# 隨機抽樣複製少數類別，直到數量跟多數類別一樣多
minority_upsampled = minority_class.sample(n=len(majority_class), replace=True, random_state=42)

# 合併
upsampled_data = pd.concat([majority_class, minority_upsampled])

# 打亂順序
upsampled_data = upsampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

X_resampled = upsampled_data.drop('Target', axis=1).values
y_resampled = upsampled_data['Target'].values

print(f"Oversampling 後的訓練集維度: {X_resampled.shape}")
print(f"Oversampling 後的類別分佈: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")

# 視覺化 Oversampling 後的分佈
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled, palette='pastel')
plt.title('Class Distribution after Oversampling (Balanced)')
plt.xticks([0, 1], ['0', '1'])
plt.savefig(os.path.join(pic_dir, '11-3_SMOTE_Distribution.png')) # 檔名維持一樣方便對照

# --- 5. 最終模型 (Final Model) ---
# 使用 SMOTE 後的資料 + 原始特徵 (這裡為了展示 SMOTE 效果，先不用 RFE 篩選過的特徵，避免變因太多)
clf_final = RandomForestClassifier(random_state=42)
clf_final.fit(X_resampled, y_resampled)
y_pred_final = clf_final.predict(X_test)

print("\n--- Final Model (with SMOTE) Results ---")
print(classification_report(y_test, y_pred_final))

# 比較 Confusion Matrix
cm_base = confusion_matrix(y_test, y_pred_base)
cm_final = confusion_matrix(y_test, y_pred_final)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Baseline Confusion Matrix')
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('SMOTE Confusion Matrix')
plt.savefig(os.path.join(pic_dir, '11-4_Confusion_Matrix_Comparison.png'))

# 儲存結果到文字檔
output_text = f"""
--- Baseline Model ---
{classification_report(y_test, y_pred_base)}

--- RFE Selected Features ---
{selected_features}

--- Final Model (with SMOTE) ---
{classification_report(y_test, y_pred_final)}
"""
with open(os.path.join(SCRIPT_DIR, 'metrics.txt'), 'w') as f:
    f.write(output_text)
