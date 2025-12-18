# Day 03: 多元線性回歸 (Multiple Linear Regression) - 創業公司利潤預測
# ---------------------------------------------------------
# 這一天的目標是處理多個特徵 (Multiple Features) 的回歸問題。
# 重點：類別特徵處理 (One-Hot Encoding) 與 虛擬變數陷阱 (Dummy Variable Trap)。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests

# 模型與評估工具
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 1. 載入資料 (Data Loading) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, '50_Startups.csv')
# 使用常見的公開資料源
DATA_URL = 'https://raw.githubusercontent.com/krishnaik06/Multiple-Linear-Regression/master/50_Startups.csv'

def load_or_download_data(local_path, url):
    if not os.path.exists(local_path):
        print(f"找不到檔案：{local_path}，嘗試從網路下載...")
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"下載成功 (requests)！已儲存為 {local_path}")
        except Exception as e:
            print(f"requests 下載失敗：{e}，嘗試使用 urllib...")
            try:
                import urllib.request
                import ssl
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                with urllib.request.urlopen(url, context=ctx) as u, open(local_path, 'wb') as f:
                    f.write(u.read())
                print(f"下載成功 (urllib)！已儲存為 {local_path}")
            except Exception as e2:
                print(f"urllib 下載也失敗：{e2}")
                return None
    else:
        print(f"發現本地檔案：{local_path}")
    return pd.read_csv(local_path)

# 嘗試讀取資料
df = load_or_download_data(DATA_FILE, DATA_URL)

if df is None:
    print("無法讀取資料，生成模擬資料...")
    # 模擬資料
    df = pd.DataFrame({
        'R&D Spend': np.random.rand(50) * 100000,
        'Administration': np.random.rand(50) * 50000,
        'Marketing Spend': np.random.rand(50) * 200000,
        'State': np.random.choice(['New York', 'California', 'Florida'], 50),
        'Profit': np.random.rand(50) * 100000 + 50000
    })

print(f"資料集維度: {df.shape}")
print(df.head())

# --- 2. 數據觀察 (EDA) ---
# 檢查數值型特徵的相關性
plt.figure(figsize=(10, 8))
# 只選取數值欄位計算相關係數
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)
plt.savefig(os.path.join(pic_dir, '3-1_Correlation.png'))
# plt.show()

# 觀察不同 State 的利潤分佈
plt.figure(figsize=(8, 6))
sns.boxplot(x='State', y='Profit', data=df)
plt.title('Profit Distribution by State')
plt.savefig(os.path.join(pic_dir, '3-2_State_Profit.png'))
# plt.show()

# --- 3. 資料分割與前處理 ---
# 處理類別變數 (One-Hot Encoding)
# drop_first=True 是為了避免虛擬變數陷阱 (Dummy Variable Trap)
# 例如：有 3 個州，我們只需要 2 個欄位 (00, 01, 10) 就能表示，不需要 3 個
df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
print("\n編碼後的資料集前五筆:")
print(df_encoded.head())

X = df_encoded.drop('Profit', axis=1)
y = df_encoded['Profit']

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化 (Standardization)
# 雖然對虛擬變數做標準化有爭議，但在混合型資料中，為了讓梯度下降收斂更快，通常還是會做
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. 建立與訓練模型 ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- 5. 模型評估 ---
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R-squared (R2): {r2:.4f}")
print(f"MSE: {mse:.4f}")

# 解析權重
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)
print("\n特徵重要性:")
print(feature_importance)

# --- 6. 結果視覺化 ---
# (A) 預測值 vs 真實值
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('True Profit')
plt.ylabel('Predicted Profit')
plt.title('True vs Predicted Profit')
plt.savefig(os.path.join(pic_dir, '3-3_Prediction.png'))
# plt.show()

# (B) 殘差圖
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Profit')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig(os.path.join(pic_dir, '3-4_Residuals.png'))
# plt.show()
