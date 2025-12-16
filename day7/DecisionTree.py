import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# 設定中文字型 (以免亂碼)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("🚀 程式開始執行...")

    # --- 1. 載入資料 ---
    filename = 'german_credit_data.csv'
    df = None

    if os.path.exists(filename):
        print(f"📂 發現本地檔案：{filename}，正在讀取...")
        try:
            # 嘗試讀取，處理可能的格式問題
            df = pd.read_csv(filename)
            # 如果讀進來只有一欄，可能是分隔符號問題 (例如是 Tab 分隔)
            if df.shape[1] < 2:
                df = pd.read_csv(filename, sep='\t')
        except Exception as e:
            print(f"⚠️ 讀取 CSV 失敗，嘗試使用 Tab 分隔: {e}")
            try:
                df = pd.read_csv(filename, sep='\t')
            except:
                pass
    
    if df is None:
        print("🌐 本地無檔案，正在嘗試從 OpenML 下載資料...")
        try:
            credit_data = fetch_openml(name='credit-g', version=1, as_frame=True)
            df = credit_data.frame
            df.to_csv(filename, index=False)
            print("✅ 下載成功！")
        except Exception as e:
            print(f"❌ 下載失敗: {e}")
            return

    print(f"📊 資料集載入完成。大小: {df.shape}")

    # --- 2. 資料前處理 (共用部分) ---
    print("⚙️ 正在進行資料前處理...")

    # 處理目標欄位
    if 'class' in df.columns:
        target_col = 'class'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        target_col = df.columns[-1]

    # 統一 Target 為 0/1 (1 = Bad/Risk, 0 = Good)
    if df[target_col].dtype == 'object':
        unique_vals = df[target_col].unique()
        if 'bad' in unique_vals:
             df['target'] = df[target_col].map({'bad': 1, 'good': 0})
        else:
             le_target = LabelEncoder()
             df['target'] = le_target.fit_transform(df[target_col])
    else:
        # 假設已經是數值，且 1/2 或 0/1
        # 若是 1/2 (1=good, 2=bad)，需轉換
        unique_vals = df[target_col].unique()
        if 2 in unique_vals and 1 in unique_vals:
             df['target'] = df[target_col].map({2: 1, 1: 0})
        else:
             df['target'] = df[target_col]

    # 移除原始 class 欄位 (避免 Data Leakage)
    if target_col != 'target':
        df = df.drop(columns=[target_col])

    # 修正: 將 category 類型轉為 object，避免 fillna 報錯
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].astype('object')

    # 填補缺失值
    df = df.fillna(0)

    # ==========================================
    # 分支 A: Decision Tree (使用 Label Encoding)
    # 適合畫出易讀的決策樹
    # ==========================================
    print("\n🌲 [A] 準備 Decision Tree 資料 (Label Encoding)...")
    df_dt = df.copy()
    encoders = {}
    for col in df_dt.columns:
        if col == 'target': continue
        if df_dt[col].dtype == 'object' or df_dt[col].dtype.name == 'category':
            le = LabelEncoder()
            df_dt[col] = le.fit_transform(df_dt[col].astype(str))
            encoders[col] = le

    X_dt = df_dt.drop('target', axis=1)
    y_dt = df_dt['target']
    X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_dt, y_dt, test_size=0.3, random_state=42)

    print("🧠 訓練 Decision Tree (Max Depth = 3)...")
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_train_dt, y_train_dt)
    
    y_pred_dt = dt_model.predict(X_test_dt)
    print(f"📊 Decision Tree 準確率: {accuracy_score(y_test_dt, y_pred_dt):.2f}")

    # ==========================================
    # 分支 B: Decision Tree (使用 One-Hot Encoding)
    # 適合追求高準確率與特徵重要性分析
    # ==========================================
    print("\n🌳 [B] 準備 Decision Tree 資料 (One-Hot Encoding)...")
    df_rf = df.copy()
    cat_cols = df_rf.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'target' in cat_cols: cat_cols.remove('target')
    
    df_rf = pd.get_dummies(df_rf, columns=cat_cols, drop_first=True)
    
    X_rf = df_rf.drop('target', axis=1)
    y_rf = df_rf['target']
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.3, random_state=42, stratify=y_rf)
    
    # 標準化
    scaler = StandardScaler()
    X_train_rf = pd.DataFrame(scaler.fit_transform(X_train_rf), columns=X_train_rf.columns)
    X_test_rf = pd.DataFrame(scaler.transform(X_test_rf), columns=X_test_rf.columns)

    print("🧠 訓練 Decision Tree (n_estimators=100)...")
    dt_model = DecisionTreeClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    dt_model.fit(X_train_rf, y_train_rf)
    
    y_pred_rf = dt_model.predict(X_test_rf)
    print(f"📊 Decision Tree 準確率: {accuracy_score(y_test_rf, y_pred_rf):.2f}")
    print(f"⭐ Decision Tree AUC: {roc_auc_score(y_test_rf, dt_model.predict_proba(X_test_rf)[:, 1]):.4f}")

    # ==========================================
    # 視覺化
    # ==========================================
    print("\n🎨 繪製圖表...")
    
    # 1. 繪製 Decision Tree
    plt.figure(figsize=(20, 10))
    class_names_list = ['Good', 'Bad'] # 0=Good, 1=Bad
    plot_tree(dt_model, 
              feature_names=X_dt.columns, 
              class_names=class_names_list,
              filled=True, rounded=True, fontsize=10)
    plt.title("Credit Risk Decision Tree (Label Encoded, Depth=3)")
    plt.show()

    # 2. 繪製 Decision Tree Feature Importance (改用決策樹的特徵重要性)
    importances = dt_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 10
    
    plt.figure(figsize=(10, 6))
    plt.title("Decision Tree Feature Importance (Top 10)")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center", color='skyblue')
    plt.xticks(range(top_n), X_dt.columns[indices[:top_n]], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print("✅ 所有程式執行完畢")

if __name__ == "__main__":
    main()