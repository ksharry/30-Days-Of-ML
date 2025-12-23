# Day 20: XGBoost (Extreme Gradient Boosting) - 糖尿病預測
# ---------------------------------------------------------
# 這一天的目標是學習 Kaggle 比賽中的王者：XGBoost。
# 它是 Gradient Boosting 的強化版，以「速度快」和「效能好」著稱。
# 我們將使用 Pima Indians Diabetes 資料集來預測糖尿病。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 嘗試匯入 xgboost，如果沒有安裝則提示
try:
    import xgboost as xgb
    from xgboost import XGBClassifier, plot_importance
except ImportError:
    print("Error: XGBoost is not installed.")
    print("Please install it using: pip install xgboost")
    exit(1)

# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 使用 sklearn 的 fetch_openml 下載 Pima Indians Diabetes Dataset
print("Downloading Pima Indians Diabetes Dataset via OpenML...")
from sklearn.datasets import fetch_openml
# version 1 is the standard UCI version
diabetes_data = fetch_openml('diabetes', version=1, as_frame=True)

df = diabetes_data.frame
print("Data loaded:")
print(f"Shape: {df.shape}")
print(df.head())

# Pima Indians Diabetes 資料集包含：
# preg: 懷孕次數
# plas: 血糖濃度
# pres: 血壓
# skin: 皮膚皺褶厚度
# insu: 胰島素濃度
# mass: BMI
# pedi: 糖尿病家族函數
# age: 年齡
# class: tested_positive (1) / tested_negative (0)

# --- 2. 資料前處理 (Preprocessing) ---
# 目標欄位處理
# OpenML 下載下來的 class 可能是 'tested_positive', 'tested_negative' 字串
# XGBoost 需要數值型的 y (0 或 1)
y = df['class'].map({'tested_positive': 1, 'tested_negative': 0})
# 如果 map 失敗 (可能是其他字串)，改用 LabelEncoder
if y.isnull().any():
    le = LabelEncoder()
    y = le.fit_transform(df['class'])

X = df.drop('class', axis=1)

# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. 訓練 XGBoost 模型 ---
# XGBoost 的強大之處在於它可以一邊訓練一邊驗證 (Early Stopping)
# n_estimators=100: 最多種 100 棵樹
# learning_rate=0.1: 學習率
# max_depth=3: 樹的深度 (避免過擬合)
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# 訓練時加入 eval_set，這樣可以看到每一輪的進步
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False # 設為 True 可以看到每一行的輸出
)

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
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix (XGBoost)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '20-1_Confusion_Matrix.png'))
print("Confusion Matrix saved.")

# --- 5. 特徵重要性 (Feature Importance) ---
# XGBoost 內建畫圖功能
plt.figure(figsize=(10, 8))
plot_importance(model, importance_type='weight', title='Feature Importance (Weight)', height=0.5)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '20-2_Feature_Importance.png'))
print("Feature Importance plot saved.")

# --- 6. 學習曲線 (Learning Curve) ---
# 我們可以從 model.evals_result() 取得訓練過程的 loss
results = model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(8, 5))
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Test')
plt.legend()
plt.ylabel('Log Loss')
plt.xlabel('Estimators (Trees)')
plt.title('XGBoost Log Loss')
plt.grid(True)
plt.savefig(os.path.join(pic_dir, '20-3_Learning_Curve.png'))
print("Learning Curve saved.")
