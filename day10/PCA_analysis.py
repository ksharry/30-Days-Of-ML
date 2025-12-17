import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 設定繪圖風格
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

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
# 將目標變數 class ('good': 0, 'bad': 1)
# 注意：原始資料欄位名稱為 'class'，且值為字串 'good'/'bad'
if 'class' in df.columns:
    df['target'] = df['class'].map({'good': 0, 'bad': 1})
    df = df.drop('class', axis=1)
else:
    print("❌ 找不到 'class' 欄位，請檢查資料集欄位名稱。")
    print(df.columns)
    exit()

# 類別型資料編碼 (Label Encoding)
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 切分特徵與標籤
X_raw = df.drop('target', axis=1)
y = df['target']

# --- PCA 核心步驟 ---

# 【重要步驟】資料標準化 (Standardization)
# PCA 對數據尺度非常敏感，必須先將所有特徵縮放到相同的範圍 (平均值=0, 標準差=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 3. 實戰：PCA 降維分析

# 3.1 視覺化：累積解釋變異量 (決定要選幾個主成分)
pca_full = PCA()
pca_full.fit(X_scaled)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_), marker='o', linestyle='--')
plt.title('PCA - Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.90, color='r', linestyle='-') # 標示 90% 變異量門檻
plt.text(1, 0.92, '90% Threshold', color = 'red', fontsize=12)
plt.grid(True)
plt.show() # 如果在本地執行請取消註解

# 3.2 視覺化：將 20 維降到 2D 平面觀察
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
pca_df_2d = pd.DataFrame(data = X_pca_2d, columns = ['PC1', 'PC2'])
pca_df_2d = pd.concat([pca_df_2d, y], axis = 1)

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='target', data=pca_df_2d, palette=['green', 'red'], alpha=0.6)
plt.title('PCA 2D Projection of German Credit Data')
plt.xlabel(f'Principal Component 1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)')
plt.show() # 如果在本地執行請取消註解


# 3.3 實戰應用：保留 90% 資訊量的降維並建模
# 設定 n_components 為 0.90，讓 PCA 自動決定需要幾個組件來保留 90% 變異量
pca_90 = PCA(n_components=0.90)
X_pca = pca_90.fit_transform(X_scaled)

print(f"\n原始特徵維度: {X_scaled.shape[1]}")
print(f"保留 90% 變異量後的維度: {X_pca.shape[1]}")
print(f"前 {X_pca.shape[1]} 個主成分總共解釋了 {np.sum(pca_90.explained_variance_ratio_)*100:.2f}% 的變異量")

# 切分訓練集與測試集 (使用降維後的資料)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# 使用簡單的邏輯回歸建模
print("\n--- Training Logistic Regression on PCA-reduced Data ---")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 預測與評估
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

print(f"\nTraining Accuracy: {acc_train:.4f}")
print(f"Test Accuracy:     {acc_test:.4f}")
print("\n--- Classification Report (Test Set) ---")
print(classification_report(y_test, y_pred_test))