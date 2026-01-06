import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

# 設定繪圖風格
plt.style.use('ggplot')

# --- 1. 載入資料 ---
filename = 'german_credit_data.csv'
try:
    # 資料集使用 Tab 分隔
    df = pd.read_csv(filename, sep='\t')
    print(f"✅ 資料載入成功！資料大小: {df.shape}")
except FileNotFoundError:
    print("❌ 找不到檔案，請確認路徑是否正確。")
    exit()

# 2. 資料預處理
# 確認目標欄位名稱 (通常是 'class' 或 'status')
target_col = 'class' if 'class' in df.columns else 'status'

# 將目標變數轉換為 0 和 1 (Good: 0, Bad: 1)
if df[target_col].dtype == 'object':
    df[target_col] = df[target_col].map({'good': 0, 'bad': 1})
else:
    df[target_col] = df[target_col].map({1: 0, 2: 1})

# 將類別型資料進行 Label Encoding
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 切分特徵與標籤
X = df.drop(target_col, axis=1)
y = df[target_col]

# 切分訓練集與測試集 (70% 訓練, 30% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 模型訓練：XGBoost
print("\n--- Training XGBoost ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    use_label_encoder=False, 
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Test Accuracy: {acc_xgb:.4f}")

# 4. 模型訓練：LightGBM
print("\n--- Training LightGBM ---")
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
acc_lgb = accuracy_score(y_test, y_pred_lgb)
print(f"LightGBM Test Accuracy: {acc_lgb:.4f}")

# 5. 比較與視覺化 (Feature Importance)
plt.figure(figsize=(15, 6))

# XGBoost Feature Importance
ax1 = plt.subplot(1, 2, 1)
xgb.plot_importance(xgb_model, ax=ax1, height=0.5, importance_type='weight', max_num_features=10, title='XGBoost Top 10 Features')

# LightGBM Feature Importance
ax2 = plt.subplot(1, 2, 2)
lgb.plot_importance(lgb_model, ax=ax2, height=0.5, importance_type='split', max_num_features=10, title='LightGBM Top 10 Features')

plt.tight_layout()

# --- 6. 混淆矩陣視覺化 ---
plt.figure(figsize=(12, 5))

# XGBoost Confusion Matrix
ax3 = plt.subplot(1, 2, 1)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=ax3)
ax3.set_title(f'XGBoost Confusion Matrix\nAcc: {acc_xgb:.4f}')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')

# LightGBM Confusion Matrix
ax4 = plt.subplot(1, 2, 2)
cm_lgb = confusion_matrix(y_test, y_pred_lgb)
sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='Greens', ax=ax4)
ax4.set_title(f'LightGBM Confusion Matrix\nAcc: {acc_lgb:.4f}')
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 輸出詳細報告 (以 XGBoost 為例)
print("\n--- XGBoost Classification Report ---")
print(classification_report(y_test, y_pred_xgb))