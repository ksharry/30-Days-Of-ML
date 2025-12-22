# Day 02: 線性回歸 (Linear Regression) - 薪資預測
# ---------------------------------------------------------
# 這一天的目標是理解最基礎的機器學習模型：線性回歸 (y = ax + b)
# 我們將使用薪資數據 (Salary_Data.csv) 來預測年資與薪水的關係。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# import requests

# 模型與評估工具
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 1. 載入資料 (Data Loading) ---
# 設定資料集路徑與下載網址
# 使用 __file__ 確保檔案路徑相對於腳本位置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'Salary_Data.csv')
DATA_URL = 'https://raw.githubusercontent.com/krishnaik06/Simple-Linear-Regression/master/Salary_Data.csv'

def load_or_download_data(local_path, url):
    """
    檢查本地是否有檔案，若無則從 URL 下載。
    """
    if not os.path.exists(local_path):
        print(f"找不到檔案：{local_path}，嘗試從網路下載...")
        try:
            import requests
            response = requests.get(url, verify=False) # 嘗試關閉 verify 避開 SSL 問題
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"下載成功 (requests)！已儲存為 {local_path}")
        except Exception as e:
            print(f"requests 下載失敗：{e}，嘗試使用 urllib...")
            try:
                import urllib.request
                import ssl
                # 忽略 SSL 憑證驗證
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
    print("無法讀取薪資資料，生成模擬資料供測試...")
    # 生成模擬資料: y = 10000 * x + 30000 + noise
    X_sim = np.random.rand(100, 1) * 10 # 0-10 年資
    y_sim = 10000 * X_sim + 30000 + np.random.randn(100, 1) * 5000
    df = pd.DataFrame(np.hstack([X_sim, y_sim]), columns=['YearsExperience', 'Salary'])

print(f"資料集維度: {df.shape}")
print(df.head())

# --- 2. 數據觀察 (EDA) ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x='YearsExperience', y='Salary', data=df)
plt.title('YearsExperience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
# 儲存圖片
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)
plt.savefig(os.path.join(pic_dir, '2-1_EDA.png'))
# plt.show()

# --- 3. 資料分割與前處理 ---
# 定義特徵 (X) 與目標 (y)
X = df[['YearsExperience']] # 保持 DataFrame 格式 (二維)
y = df['Salary']

# 切分資料 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化 (Standardization) - 雖然簡單線性回歸不一定需要，但養成好習慣
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

# 獲取回歸係數 (注意：這是針對標準化後數據的係數)
print(f"Intercept (b): {model.intercept_:.2f}")
print(f"Coefficient (a): {model.coef_[0]:.2f}")

# --- 6. 結果視覺化 ---
# (A) 預測結果圖
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
# 因為 X_test_scaled 是標準化過的，畫線時要用原始 X_test 來對應預測值，或者直接用模型預測
# 為了畫出平滑的線，我們生成一系列點
X_range = np.linspace(X['YearsExperience'].min(), X['YearsExperience'].max(), 100).reshape(-1, 1)
X_range_scaled = scaler.transform(X_range)
y_range_pred = model.predict(X_range_scaled)

plt.plot(X_range, y_range_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Salary Prediction (Linear Regression)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.savefig(os.path.join(pic_dir, '2-2_Prediction.png'))
# plt.show()

# (B) 殘差圖
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Salary')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig(os.path.join(pic_dir, '2-3_Residuals.png'))
# plt.show()
