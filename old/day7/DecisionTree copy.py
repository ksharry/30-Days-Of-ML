import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("🚀 程式開始執行...")

    # --- 1. 載入資料 ---
    filename = 'german_credit_data.csv'
    df = None

    if os.path.exists(filename):
        print(f"📂 發現本地檔案：{filename}，正在讀取...")
        try:
            df = pd.read_csv(filename, sep='\t') # 注意：如果是 OpenML 下載的通常是逗號，如果是以前的可能是 Tab
        except:
            # 如果 Tab 讀失敗，嘗試用逗號
            df = pd.read_csv(filename, sep=',')
    
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

    # --- 2. 資料前處理 ---
    print("⚙️ 正在進行資料前處理...")

    # 2.1 處理目標欄位 (class)
    # 確保目標欄位名稱統一
    if 'class' in df.columns:
        target_col = 'class'
    else:
        # 假設最後一欄是目標
        target_col = df.columns[-1]

    # 將 bad/good 轉為 1/0
    # 這裡我們手動定義 class_names 方便等一下畫圖
    class_names_list = ['Good', 'Bad'] # 0對應Good, 1對應Bad
    
    # 簡單的映射：假設 bad 是風險 (1), good 是安全 (0)
    # 如果資料已經是數字，這行可能需要調整，但 OpenML 預設是字串
    if df[target_col].dtype == 'object':
        df['target'] = df[target_col].map({'bad': 1, 'good': 0})
    else:
        df['target'] = df[target_col] # 假設已經是數字
        
    # 移除原始 class 欄位
    if target_col != 'target':
        df = df.drop(columns=[target_col])

    # 2.2 處理特徵 (Label Encoding)
    # 為了畫出漂亮的樹，我們使用 LabelEncoder 而不是 One-Hot
    encoders = {} # 這裡定義 encoders 字典，解決您的 NameError
    
    for col in df.columns:
        if col == 'target': continue
        
        # 如果是文字欄位，轉成數字
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le # 存起來，畫圖時用

    # --- 3. 切分資料 ---
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 補缺失值 (以防萬一)
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- 4. 訓練模型 (Decision Tree) ---
    print("🧠 開始訓練決策樹 (Decision Tree)...")
    # max_depth=3 限制樹的深度，讓圖看得清楚
    model = DecisionTreeClassifier(max_depth=3, random_state=42) 
    model.fit(X_train, y_train)

    # --- 5. 評估結果 ---
    y_pred = model.predict(X_test)
    print(f"\n📊 模型準確率: {accuracy_score(y_test, y_pred):.2f}")
    print("\n分類報告:")
    print(classification_report(y_test, y_pred, target_names=class_names_list))

    # --- 6. 畫出決策樹 ---
    print("🎨 正在繪製決策樹...")
    plt.figure(figsize=(20, 10))
    
    plot_tree(model, 
              feature_names=X.columns, 
              class_names=class_names_list, # 使用我們定義好的 ['Good', 'Bad']
              filled=True, 
              rounded=True, 
              fontsize=10)
    
    plt.title("Credit Risk Decision Tree (Max Depth = 3)")
    plt.show() # 視窗應該會跳出來
    print("✅ 程式執行完畢")

if __name__ == "__main__":
    main()