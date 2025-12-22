# Day 12: 分類模型評估 (Model Evaluation) - 鐵達尼號生存預測
# ---------------------------------------------------------
# 這一天的目標是深入理解分類模型的成績單。
# 我們沿用鐵達尼號資料集，但這次重點不在模型本身，而在於如何「全方位」評估它。
# 重點：混淆矩陣, Precision/Recall/F1, ROC/AUC, K-Fold Cross Validation。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 模型與評估工具
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. 載入與預處理資料 (Data Loading & Preprocessing) ---
# 為了方便，我們直接使用簡單的預處理邏輯 (類似 Day 6/9)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(os.path.dirname(SCRIPT_DIR), 'day09', 'titanic.csv') # 借用 Day 9 的資料

if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found. Please ensure day9/titanic.csv exists.")
    exit()

df = pd.read_csv(DATA_FILE)

# 簡單清洗
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# 編碼
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

X = df.drop('Survived', axis=1)
y = df['Survived']

# 標準化 (對 Logistic Regression 很重要)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# --- 2. 訓練模型 (Model Training) ---
# 使用邏輯回歸作為示範模型
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 預測機率 (而不只是類別)，這對 ROC/AUC 很重要
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # 只要 "1" (生存) 的機率

# --- 3. 評估指標視覺化 (Evaluation Metrics Visualization) ---
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 3.1 混淆矩陣 (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(pic_dir, '12-1_Confusion_Matrix.png'))

# 3.2 ROC 曲線 (Receiver Operating Characteristic Curve)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(pic_dir, '12-2_ROC_Curve.png'))

# 3.3 Precision-Recall 曲線
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig(os.path.join(pic_dir, '12-3_PR_Curve.png'))

# --- 4. 交叉驗證 (Cross Validation) ---
# 這是評估模型穩健性最重要的步驟
# 我們把資料切成 10 份 (K=10)，輪流當測試集，看平均分數
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), cv_scores, marker='o', linestyle='--', color='purple')
plt.axhline(y=cv_scores.mean(), color='r', linestyle='-', label=f'Mean Accuracy: {cv_scores.mean():.4f}')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('10-Fold Cross Validation Scores')
plt.legend()
plt.ylim([0.7, 0.9]) # 設定範圍讓波動更明顯
plt.savefig(os.path.join(pic_dir, '12-4_Cross_Validation.png'))

# --- 5. 輸出報表 ---
output_text = f"""
--- Classification Report ---
{classification_report(y_test, y_pred)}

--- ROC AUC Score ---
{roc_auc:.4f}

--- Cross Validation Scores (10-Fold) ---
Scores: {cv_scores}
Mean Accuracy: {cv_scores.mean():.4f}
Standard Deviation: {cv_scores.std():.4f}
"""

with open(os.path.join(SCRIPT_DIR, 'metrics.txt'), 'w') as f:
    f.write(output_text)

print("Evaluation complete. Check metrics.txt and pic/ folder.")
