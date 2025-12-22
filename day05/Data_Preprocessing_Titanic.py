# Day 05: 數據預處理 (Data Preprocessing) - 鐵達尼號生存預測
# ---------------------------------------------------------
# 這一天的目標是建立機器學習的穩固地基。
# 我們將不進行模型訓練，而是專注於將「髒亂的原始數據」轉化為「模型可讀的乾淨數據」。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests

# 預處理工具
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- 1. 載入資料 (Data Loading) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'titanic.csv')
DATA_URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

def load_or_download_data(local_path, url):
    if not os.path.exists(local_path):
        print(f"找不到檔案：{local_path}，嘗試從網路下載...")
        try:
            response = requests.get(url, verify=False)
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

df = load_or_download_data(DATA_FILE, DATA_URL)
if df is None:
    # Fallback if download fails
    df = sns.load_dataset('titanic')

print(f"原始資料集維度: {df.shape}")
print(df.head())

# --- 2. 數據觀察 (EDA) - 偵測髒數據 ---
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# (A) 缺失值熱力圖 (Missing Values Heatmap)
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap (Yellow = Missing)')
plt.savefig(os.path.join(pic_dir, '5-1_Missing_Values.png'))
# plt.show()

# 計算缺失值比例
missing_percent = df.isnull().sum() / len(df) * 100
print("\n缺失值比例 (%):")
print(missing_percent[missing_percent > 0])

# --- 3. 數據預處理 (Preprocessing) ---

# 步驟 1: 特徵篩選 (Feature Selection)
# 剔除對預測顯然無用的欄位 (如名字、票號、客艙編號-缺太多)
# PassengerId, Name, Ticket, Cabin
selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
df_clean = df[selected_features].copy()

# 分離 X 和 y
X = df_clean.drop('Survived', axis=1)
y = df_clean['Survived']

# 定義數值型與類別型特徵
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# 步驟 2: 建立預處理流水線 (Pipeline)
# 這是 Scikit-Learn 最強大的功能，將填補、編碼、縮放串接起來

# 數值型處理：填補缺失值 (平均數) -> 標準化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # 年齡缺值補平均
    ('scaler', StandardScaler())                 # 數值標準化
])

# 類別型處理：填補缺失值 (最頻數) -> One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # 登船港口缺值補眾數
    ('encoder', OneHotEncoder(drop='first'))              # 轉為 0/1，drop='first' 避開虛擬變數陷阱
])

# 組合處理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 步驟 3: 切分資料 (Train/Test Split)
# 注意：必須先切分，再 fit 預處理器，避免 Data Leakage (測試集資訊洩漏到訓練集)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步驟 4: 執行預處理
# 在訓練集上 fit_transform
X_train_processed = preprocessor.fit_transform(X_train)
# 在測試集上 transform (不能 fit!)
X_test_processed = preprocessor.transform(X_test)

# 取得處理後的欄位名稱 (為了視覺化)
# OneHotEncoder 會產生新欄位名
cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_feature_names)

# 轉回 DataFrame 以便觀察
X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)

print("\n預處理後的訓練集前五筆:")
print(X_train_df.head())

# --- 4. 結果視覺化 (Before vs After) ---

# (B) 數據分佈比較 (以 Age 為例)
plt.figure(figsize=(12, 5))

# Before: 原始資料 (有缺失值，未標準化)
plt.subplot(1, 2, 1)
sns.histplot(X_train['Age'].dropna(), kde=True, color='blue')
plt.title('Before: Age Distribution (Original)')

# After: 處理後資料 (已填補，已標準化)
plt.subplot(1, 2, 2)
sns.histplot(X_train_df['Age'], kde=True, color='green')
plt.title('After: Age Distribution (Scaled & Imputed)')

plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '5-2_Distribution_Compare.png'))
# plt.show()

print(f"\n處理完成！圖片已儲存至 {pic_dir}")
