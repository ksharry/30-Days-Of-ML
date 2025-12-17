import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def main():
    print("🚀 Day 8: 隨機森林 (Random Forest) 啟動...")

    # --- 1. 載入資料 ---
    filename = 'german_credit_data.csv'
    try:
        # 嘗試讀取，處理分隔符號問題
        df = pd.read_csv(filename, sep='\t')
        if len(df.columns) < 5: 
            df = pd.read_csv(filename, sep=',')
        print(f"✅ 資料載入成功！資料大小: {df.shape}")
    except FileNotFoundError:
        print("❌ 找不到檔案，請確認路徑是否正確。")
        return

    # --- 2. 資料前處理 ---
    # 2.1 處理 Target
    if 'class' in df.columns:
        target_col = 'class'
    else:
        target_col = df.columns[-1]

    # 轉成 0 (Good) 和 1 (Bad/Risk)
    if df[target_col].dtype == 'object':
        df['target'] = df[target_col].map({'bad': 1, 'good': 0})
    else:
        df['target'] = df[target_col]
    
    if target_col != 'target':
        df = df.drop(columns=[target_col])

    # 2.2 特徵編碼 (Label Encoding)
    # 隨機森林雖然強大，但 sklearn 版本仍建議將文字轉數字
    for col in df.columns:
        if col == 'target': continue
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # --- 3. 切分資料 ---
    X = df.drop('target', axis=1)
    y = df['target']
    X = X.fillna(0) # 簡易補值

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # --- 4. 建立並訓練隨機森林 ---
    print("🌲 正在種植 100 棵決策樹 (Training)...")
    # n_estimators=100: 種 100 棵樹
    # max_depth=None: 不限制深度，讓樹自由生長 (隨機森林不怕過擬合)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # --- 5. 評估與視覺化 ---
    # 5.1 準確率比較 (Overfitting Check)
    train_acc = rf_model.score(X_train, y_train)
    print(f"\n🎯 訓練集準確率 (Training Acc): {train_acc:.2f}")

    y_pred = rf_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"🏆 測試集準確率 (Test Acc):     {test_acc:.2f}")
    
    # 設定畫布
    plt.figure(figsize=(14, 6))
    
    # [左圖] 混淆矩陣
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
    plt.title('Confusion Matrix (Prediction Accuracy)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # [右圖] 特徵重要性 (Feature Importance)
    plt.subplot(1, 2, 2)
    importances = rf_model.feature_importances_
    # 排序取得前 10 名
    indices = np.argsort(importances)[::-1]
    top_n = 10
    
    plt.title('Top 10 Key Features (What matters most?)')
    plt.barh(range(top_n), importances[indices[:top_n]][::-1], color='forestgreen', align='center')
    plt.yticks(range(top_n), X.columns[indices[:top_n]][::-1])
    plt.xlabel('Importance Score')
    
    plt.tight_layout()
    plt.show()
    print("✅ 分析完成！右圖顯示了影響信用評分最關鍵的因素。")

if __name__ == "__main__":
    main()