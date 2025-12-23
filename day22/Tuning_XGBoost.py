# Day 22: Hyperparameter Tuning - 模型調參 (XGBoost)
# ---------------------------------------------------------
# 這一天的目標是學習如何幫模型「調音」(Tuning)。
# 模型就像一台精密的儀器，有很多旋鈕 (Hyperparameters) 可以轉。
# 我們將比較兩種最常見的調參方法：
# 1. Grid Search (網格搜索)：地毯式搜索，慢但精確。
# 2. Random Search (隨機搜索)：隨機嘗試，快且效果通常不錯。
# 我們沿用 Day 20 的 Pima Indians Diabetes 資料集。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
from xgboost import XGBClassifier

# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

print("Downloading Pima Indians Diabetes Dataset via OpenML...")
diabetes_data = fetch_openml('diabetes', version=1, as_frame=True)
df = diabetes_data.frame

# 處理標籤
y = df['class'].map({'tested_positive': 1, 'tested_negative': 0})
if y.isnull().any():
    le = LabelEncoder()
    y = le.fit_transform(df['class'])
X = df.drop('class', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- 2. 設定參數範圍 (Parameter Grid) ---
# 我們要調整 XGBoost 的三個重要參數：
# 1. n_estimators: 樹的數量 (打幾桿/改幾次考卷)
# 2. learning_rate: 學習率 (修正幅度)
# 3. max_depth: 樹的深度 (模型複雜度)

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
# 總共組合數：3 * 3 * 3 = 27 種組合

# 建立基礎模型
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# --- 3. Grid Search (網格搜索) ---
print("\nRunning Grid Search (Trying all 27 combinations)...")
start_time = time.time()

# cv=3: 3-Fold Cross Validation (每組參數測 3 次取平均)
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train, y_train)

grid_time = time.time() - start_time
print(f"Grid Search Time: {grid_time:.2f} seconds")
print(f"Best Params (Grid): {grid_search.best_params_}")
print(f"Best Score (Grid): {grid_search.best_score_:.4f}")

# --- 4. Random Search (隨機搜索) ---
print("\nRunning Random Search (Trying 10 random combinations)...")
start_time = time.time()

# n_iter=10: 只隨機抽 10 組參數來測 (原本是 27 組)
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid, n_iter=10, cv=3, scoring='accuracy', n_jobs=1, random_state=42)
random_search.fit(X_train, y_train)

random_time = time.time() - start_time
print(f"Random Search Time: {random_time:.2f} seconds")
print(f"Best Params (Random): {random_search.best_params_}")
print(f"Best Score (Random): {random_search.best_score_:.4f}")

# --- 5. 驗證與比較 (Validation & Comparison) ---
# 使用找到的最佳參數來預測測試集
best_grid_model = grid_search.best_estimator_
best_random_model = random_search.best_estimator_

y_pred_grid = best_grid_model.predict(X_test)
y_pred_random = best_random_model.predict(X_test)

acc_grid = accuracy_score(y_test, y_pred_grid)
acc_random = accuracy_score(y_test, y_pred_random)

print(f"\n--- Final Test Set Evaluation ---")
print(f"Grid Search Test Accuracy: {acc_grid:.4f}")
print(f"Random Search Test Accuracy: {acc_random:.4f}")

# --- 6. 視覺化：時間 vs 效能 (Time vs Performance) ---
methods = ['Grid Search', 'Random Search']
times = [grid_time, random_time]
accuracies = [acc_grid, acc_random]

fig, ax1 = plt.subplots(figsize=(8, 6))

# 畫時間 (Bar Chart)
color = 'tab:blue'
ax1.set_xlabel('Method')
ax1.set_ylabel('Time (seconds)', color=color)
ax1.bar(methods, times, color=color, alpha=0.6, width=0.4)
ax1.tick_params(axis='y', labelcolor=color)

# 畫準確率 (Line Chart)
ax2 = ax1.twinx()  # 共用 x 軸
color = 'tab:red'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(methods, accuracies, color=color, marker='o', linewidth=2, markersize=10)
ax2.tick_params(axis='y', labelcolor=color)
# 設定 y 軸範圍讓差異明顯一點
ax2.set_ylim(min(accuracies)*0.95, max(accuracies)*1.02)

plt.title('Grid Search vs Random Search: Time & Accuracy')
plt.savefig(os.path.join(pic_dir, '22-1_Tuning_Comparison.png'))
print("Comparison plot saved.")
