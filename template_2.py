# 模版二：監督式分類 (Supervised Classification)
# 適用日子：Day 06-12 (邏輯回歸, KNN, SVM, 決策樹), Day 18-22 (隨機森林, XGBoost)
# ==========================================
# Day XX: [填入演算法名稱，如 Logistic Regression]
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests

# 模型與評估工具
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 根據 Day 換模型：
from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

# --- 1. 載入資料 (Data Loading) ---
# 設定資料集路徑與下載網址 (請依實際需求修改)
DATA_FILE = 'dataset.csv' 
DATA_URL = 'https://raw.githubusercontent.com/ksharry/30-Days-Of-ML/main/dayXX/dataset.csv'

def load_or_download_data(local_path, url):
    """
    檢查本地是否有檔案，若無則從 URL 下載。
    """
    if not os.path.exists(local_path):
        print(f"找不到檔案：{local_path}，嘗試從網路下載...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"下載成功！已儲存為 {local_path}")
        except Exception as e:
            print(f"下載失敗：{e}")
            return None
    else:
        print(f"發現本地檔案：{local_path}")
    
    return pd.read_csv(local_path)

# 嘗試讀取資料
try:
    df = load_or_download_data(DATA_FILE, DATA_URL)
    if df is None:
        raise ValueError("無法讀取資料")
except Exception as e:
    print(f"讀取自定義資料失敗 ({e})，改用 Scikit-Learn 內建乳癌資料集範例...")
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Target'] = data.target

print(f"資料集維度: {df.shape}")
print(df.head())

# --- 2. 數據觀察 (EDA) ---
# 檢查類別分佈
sns.countplot(x='Target', data=df)
plt.title('Class Distribution')
plt.show()

# --- 3. 資料分割與前處理 ---
X = df.drop('Target', axis=1)
y = df['Target']

# 切分資料 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化 (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. 建立與訓練模型 ---
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# --- 5. 模型評估 ---
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 6. 結果視覺化 ---
# 混淆矩陣熱力圖
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 若是二維特徵，可繪製決策邊界 (Decision Boundary)
# (此處略，視資料維度而定)