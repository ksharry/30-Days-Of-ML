# Day 21: Stacking (Stacked Generalization) - 心臟病預測
# ---------------------------------------------------------
# 這一天的目標是學習集成學習的終極大招：Stacking (堆疊法)。
# 它的概念是：集合各路專家 (KNN, SVM, RF) 的意見，
# 再請一位「總醫師」 (Meta Learner) 來做最終判斷。
# 我們將使用 Heart Disease 資料集來實作。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import fetch_openml

# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

print("Downloading Heart Disease Dataset (Statlog) via OpenML...")
# Statlog (Heart) Data Set: 預測心臟病是否存在
# Target: 1 (Absence), 2 (Presence) -> 我們會轉成 0 和 1
data = fetch_openml('heart-statlog', version=1, as_frame=True)
X = data.data
y = data.target

# 轉換標籤：1 -> 0 (健康), 2 -> 1 (有病)
le = LabelEncoder()
y = le.fit_transform(y) 

print("Data loaded:")
print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")

# 因為要用 KNN 和 SVM，特徵縮放 (Scaling) 很重要
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 2. 設定基底模型 (Base Learners) ---
# 我們選三個截然不同的模型，希望能有「互補」的效果
# 1. KNN: 看鄰居 (幾何距離)
# 2. SVM: 畫界線 (幾何邊界)
# 3. Random Forest: 問問題 (邏輯規則)
base_learners = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42))
]

# --- 3. 設定元模型 (Meta Learner) ---
# 通常用簡單的模型，如 Logistic Regression
meta_learner = LogisticRegression()

# --- 4. 建立 Stacking 模型 ---
# cv=5: 內部使用 5-fold cross validation 來產生給 meta learner 的訓練資料
stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)

# --- 5. 訓練與評估 ---
print("\nTraining models...")
models = {
    'KNN': base_learners[0][1],
    'SVM': base_learners[1][1],
    'Random Forest': base_learners[2][1],
    'Stacking': stacking_model
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# --- 6. 視覺化：效能比較 (Performance Comparison) ---
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
plt.ylim(0.7, 1.0) # 設定 Y 軸範圍讓差異更明顯
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(pic_dir, '21-1_Accuracy_Comparison.png'))
print("Accuracy Comparison plot saved.")

# --- 7. 視覺化：模型相關性 (Correlation of Predictions) ---
# Stacking 有效的前提是：基底模型要有「多樣性」 (大家錯的地方不一樣)
# 我們來看看這三個基底模型的預測結果像不像
predictions = pd.DataFrame()
for name, model in models.items():
    if name != 'Stacking':
        predictions[name] = model.predict(X_test)

plt.figure(figsize=(8, 6))
sns.heatmap(predictions.corr(), annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Correlation of Base Learners Predictions')
plt.savefig(os.path.join(pic_dir, '21-2_Prediction_Correlation.png'))
print("Correlation Heatmap saved.")

# --- 8. 視覺化：混淆矩陣 (Stacking) ---
y_pred_stack = stacking_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_stack)

# 印出數值供 README 使用
tn, fp, fn, tp = cm.ravel()
print(f"\n--- Confusion Matrix Values (Stacking) ---")
print(f"TN (True Negative): {tn}")
print(f"FP (False Positive): {fp}")
print(f"FN (False Negative): {fn}")
print(f"TP (True Positive): {tp}")
print(f"Total: {tn + fp + fn + tp}")
print("------------------------------------------\n")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Heart Disease'], yticklabels=['Healthy', 'Heart Disease'])
plt.title('Confusion Matrix (Stacking)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(pic_dir, '21-3_Confusion_Matrix.png'))
print("Confusion Matrix saved.")
