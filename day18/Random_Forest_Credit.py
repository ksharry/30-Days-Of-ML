# Day 18: 隨機森林 (Random Forest) - 信用貸款風險預測
# ---------------------------------------------------------
# 這一天的目標是學習「集成學習 (Ensemble Learning)」中的 Bagging 技術。
# 隨機森林由許多決策樹組成，透過「眾人智慧」來解決單一決策樹容易過擬合的問題。
# 我們將使用德國信用風險資料集 (German Credit Data) 來預測客戶是否會違約。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 使用 sklearn 的 fetch_openml 下載 German Credit Dataset
# 這比直接處理 raw text file 方便很多
print("Downloading German Credit Dataset via OpenML...")
from sklearn.datasets import fetch_openml
credit_data = fetch_openml('credit-g', version=1, as_frame=True)

df = credit_data.frame
print("Data loaded:")
print(f"Shape: {df.shape}")
print(df.head())

# --- 2. 資料前處理 (Preprocessing) ---
# 這個資料集有很多類別型特徵 (Categorical Features)，例如 'checking_status', 'purpose'
# 隨機森林雖然強大，但 sklearn 的實作需要我們把文字轉成數字

# 簡單起見，我們使用 LabelEncoder 把所有 object 類型的欄位轉成數字
# (嚴謹的做法應該用 OneHotEncoder，但為了保持程式碼簡潔，這裡用 LabelEncoder)
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        df[col] = le.fit_transform(df[col])

# 定義特徵 (X) 和目標 (y)
# 目標欄位是 'class' (good=1, bad=0 after encoding, let's check)
# 原本 class 是 'good', 'bad'。LabelEncoder 會轉成 0, 1。
# 我們確認一下哪個是哪個
print("\nClass distribution:")
print(df['class'].value_counts())

X = df.drop('class', axis=1)
y = df['class']

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. 訓練隨機森林模型 (Random Forest Training) ---
# n_estimators=100: 森林裡有 100 棵樹
# random_state=42: 固定隨機種子
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- 4. 模型評估 (Evaluation) ---
y_pred = rf_model.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 繪製混淆矩陣
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.title('Confusion Matrix (Random Forest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '18-1_Confusion_Matrix.png'))
print("Confusion Matrix saved.")

# --- 5. 特徵重要性 (Feature Importance) ---
# 這是隨機森林最強大的功能之一：告訴我們哪些特徵最重要！
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1] # 由大到小排序
feature_names = X.columns

print("\nTop 5 Important Features:")
for i in range(5):
    print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")

# 繪製特徵重要性圖
plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Credit Risk)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '18-2_Feature_Importance.png'))
print("Feature Importance plot saved.")

# --- 6. 視覺化其中一棵樹 (Visualize Single Tree) ---
# 隨機森林有 100 棵樹，我們畫出其中一棵來看看它長什麼樣子
# 限制深度 max_depth=3 以免圖太大看不清楚
from sklearn.tree import plot_tree

plt.figure(figsize=(15, 10))
# 取出森林中的第一棵樹 (estimators_[0])
plot_tree(rf_model.estimators_[0], 
          feature_names=feature_names,
          class_names=['Bad', 'Good'],
          filled=True, 
          max_depth=3, 
          fontsize=10)
plt.title('Single Decision Tree from Random Forest (Depth=3)')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '18-3_Single_Tree.png'))
print("Single Tree visualization saved.")
