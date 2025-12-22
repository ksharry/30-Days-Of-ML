# Day 06: 邏輯回歸 (Logistic Regression) - 社交網絡廣告分類
# ---------------------------------------------------------
# 這一天的目標是進入「分類 (Classification)」的世界。
# 雖然名字叫回歸，但它是用於分類的！
# 重點：Sigmoid 函數、決策邊界 (Decision Boundary)、混淆矩陣。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from matplotlib.colors import ListedColormap

# 模型與評估工具
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# ... (略)



# --- 1. 載入資料 (Data Loading) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'Social_Network_Ads.csv')
# 使用常見的公開資料源
DATA_URL = 'https://raw.githubusercontent.com/shivang98/Social-Network-ads-Boost/master/Social_Network_Ads.csv'

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
    print("無法讀取資料，生成模擬資料...")
    # 模擬資料
    df = pd.DataFrame({
        'Age': np.random.randint(18, 60, 400),
        'EstimatedSalary': np.random.randint(15000, 150000, 400),
        'Purchased': np.random.randint(0, 2, 400)
    })

print(f"資料集維度: {df.shape}")
print(df.head())

# --- 2. 數據觀察 (EDA) ---
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 觀察 Age 與 Salary 對購買行為的影響
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='EstimatedSalary', hue='Purchased', data=df, palette='winter')
plt.title('Age vs Estimated Salary (Purchased?)')
plt.savefig(os.path.join(pic_dir, '6-1_EDA_Scatter.png'))
# plt.show()

# --- 3. 資料分割與前處理 ---
# 我們只選取 Age 和 EstimatedSalary 作為特徵，方便繪製 2D 決策邊界
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 標準化 (Standardization) - 對邏輯回歸非常重要！
# 因為它使用梯度下降優化，且繪製決策邊界時需要統一尺度
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. 建立與訓練模型 ---
model = LogisticRegression(random_state=0)
model.fit(X_train_scaled, y_train)

# --- 5. 模型評估 ---
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # 取得預測為 1 (購買) 的機率，用於計算 AUC

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_output = f"""
Accuracy: {acc:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1:.4f}
AUC: {auc:.4f}
Confusion Matrix:
{cm}

Classification Report:
{classification_report(y_test, y_pred)}
"""

print(metrics_output)
with open(os.path.join(SCRIPT_DIR, 'metrics.txt'), 'w') as f:
    f.write(metrics_output)

# 繪製混淆矩陣
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(pic_dir, '6-2_Confusion_Matrix.png'))
# plt.show()

# 繪製 ROC 曲線 (新增)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.savefig(os.path.join(pic_dir, '6-4_ROC_Curve.png'))
# plt.show()

# --- 6. 結果視覺化 (決策邊界 Decision Boundary) ---
# 這段程式碼比較複雜，用於畫出模型分類的「界線」
def plot_decision_boundary(X_set, y_set, title, filename):
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    
    plt.figure(figsize=(10, 6))
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j, edgecolor='black')
        
    plt.title(title)
    plt.xlabel('Age (Scaled)')
    plt.ylabel('Estimated Salary (Scaled)')
    plt.legend()
    plt.savefig(os.path.join(pic_dir, filename))
    # plt.show()

# 繪製測試集的決策邊界
plot_decision_boundary(X_test_scaled, y_test, 'Logistic Regression (Test set)', '6-3_Decision_Boundary.png')
