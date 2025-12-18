import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("🧪 Day 8 進階實驗室：隨機森林自動調參 (Hyperparameter Tuning)")
    print("========================================================")

    # --- 1. 載入與處理資料 (與之前相同) ---
    filename = 'german_credit_data.csv'
    try:
        df = pd.read_csv(filename, sep='\t')
        if len(df.columns) < 5: df = pd.read_csv(filename, sep=',')
    except FileNotFoundError:
        print("❌ 找不到檔案")
        return

    # 處理 Target
    target_col = 'class' if 'class' in df.columns else df.columns[-1]
    if df[target_col].dtype == 'object':
        df['target'] = df[target_col].map({'bad': 1, 'good': 0})
    else:
        df['target'] = df[target_col]
    if target_col != 'target': df = df.drop(columns=[target_col])

    # 處理特徵 (Label Encoding)
    for col in df.columns:
        if col == 'target': continue
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # 切分資料
    X = df.drop('target', axis=1).fillna(0)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"✅ 資料準備完成，訓練集數量: {len(X_train)}，測試集數量: {len(X_test)}")
    print("-" * 60)

    # --- 2. Round 1: 原始模型 (Default) ---
    print("🥊 Round 1: 使用預設參數 (Default) 訓練中...")
    default_model = RandomForestClassifier(random_state=42)
    default_model.fit(X_train, y_train)
    
    default_acc = accuracy_score(y_test, default_model.predict(X_test))
    print(f"👉 原始模型準確率: {default_acc:.4f}")
    print("-" * 60)

    # --- 3. Round 2: 自動調參 (Grid Search) ---
    print("🥊 Round 2: 啟動 GridSearchCV 自動調參...")
    print("   (這會測試多種組合，請稍候...)")

    # 設定參數網格 (您可以試著修改這裡的數值)
    param_grid = {
        'n_estimators': [50, 100, 200],        # 樹的數量
        'max_depth': [10, 20, None],           # 樹的深度限制
        'min_samples_split': [2, 5],           # 節點再切分的最少樣本數
        'class_weight': ['balanced', None]     # 是否加重壞人權重
    }

    # 建立搜尋器
    # cv=5: 做 5 次交叉驗證 (Cross Validation)
    # n_jobs=-1: 用盡電腦所有 CPU 核心去跑
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=5, n_jobs=-1, verbose=1)
    
    # 開始計時並訓練
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()

    print(f"\n✅ 搜尋完成！耗時: {end_time - start_time:.2f} 秒")
    print(f"🔍 總共測試了 {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['class_weight'])} 種組合")
    print("-" * 60)

    # --- 4. 結果分析與 PK ---
    best_model = grid_search.best_estimator_
    best_acc = accuracy_score(y_test, best_model.predict(X_test))

    print("🏆 調參結果報告")
    print(f"👑 最佳參數組合: {grid_search.best_params_}")
    print(f"📈 最佳模型準確率: {best_acc:.4f}")
    
    improvement = (best_acc - default_acc) * 100
    print("-" * 60)
    if improvement > 0:
        print(f"🎉 恭喜！經過調參，模型準確率提升了 {improvement:.2f}%")
    elif improvement == 0:
        print(f"😐 持平。看來預設參數已經很強了，或者是資料量的限制。")
    else:
        print(f"📉 微幅下降。這在測試集上偶爾會發生，代表最佳參數在訓練集雖強，但在測試集稍弱(過擬合風險)。")

    # 顯示詳細分類報告 (特別看 Recall 是否有提升)
    print("\n📄 最佳模型的詳細報告:")
    print(classification_report(y_test, best_model.predict(X_test), target_names=['Good', 'Bad']))

if __name__ == "__main__":
    main()