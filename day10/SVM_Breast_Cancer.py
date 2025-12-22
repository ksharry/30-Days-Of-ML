# Day 10: 支持向量機 (SVM) - 乳癌檢測
# ---------------------------------------------------------
# 這一天的目標是學習最強大的傳統分類器：SVM。
# 我們使用 Breast Cancer Wisconsin 資料集。
# 重點：超平面 (Hyperplane), 核函數 (Kernel Trick), 邊際 (Margin)。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import ListedColormap

# 模型與評估工具
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.decomposition import PCA

# --- 1. 載入資料 (Data Loading) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 使用 sklearn 內建的乳癌資料集
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# 轉為 DataFrame 方便觀察
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"資料集維度: {df.shape}")
print(df.head())
print(f"類別分佈: {dict(zip(*np.unique(y, return_counts=True)))}")
# 0: Malignant (惡性), 1: Benign (良性)

# --- 2. 資料分割與前處理 ---
# SVM 對特徵縮放非常敏感，一定要做標準化！
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. 建立與訓練模型 ---
# 我們嘗試兩種 Kernel：
# 1. Linear (線性核): 畫一條直線切分
# 2. RBF (徑向基函數核): 畫一個圈圈或不規則形狀切分 (預設)

# 這裡先用 Linear Kernel 來展示基本概念
classifier_linear = SVC(kernel='linear', random_state=0)
classifier_linear.fit(X_train_scaled, y_train)

# 再用 RBF Kernel 來展示非線性能力
classifier_rbf = SVC(kernel='rbf', random_state=0)
classifier_rbf.fit(X_train_scaled, y_train)

# --- 4. 模型評估 ---
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"--- {name} Results ---")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 30)
    return cm, acc

cm_linear, acc_linear = evaluate_model(classifier_linear, X_test_scaled, y_test, "Linear Kernel")
cm_rbf, acc_rbf = evaluate_model(classifier_rbf, X_test_scaled, y_test, "RBF Kernel")

metrics_output = f"""
Linear Kernel Accuracy: {acc_linear:.4f}
RBF Kernel Accuracy: {acc_rbf:.4f}

Linear Confusion Matrix:
{cm_linear}

RBF Confusion Matrix:
{cm_rbf}
"""

with open(os.path.join(SCRIPT_DIR, 'metrics.txt'), 'w') as f:
    f.write(metrics_output)

# 繪製混淆矩陣 (以 RBF 為例)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix (SVM - RBF Kernel)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(pic_dir, '10-1_Confusion_Matrix.png'))

# --- 5. 結果視覺化 (決策邊界) ---
# 因為乳癌資料集有 30 個特徵，無法直接畫 2D 圖。
# 我們使用 PCA (主成分分析) 將 30 維降到 2 維，再來畫 SVM 的邊界。
# 注意：這只是為了視覺化，實際訓練還是用 30 維效果最好。

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 重新訓練一個 2D 的 SVM 模型用於畫圖
classifier_viz = SVC(kernel='rbf', random_state=0)
classifier_viz.fit(X_train_pca, y_train)

def plot_decision_boundary(X_set, y_set, model, title, filename):
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    
    plt.figure(figsize=(10, 6))
    Z = model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(('red', 'green')))
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=target_names[j], edgecolor='black')
        
    plt.title(title)
    plt.xlabel('PC1 (Principal Component 1)')
    plt.ylabel('PC2 (Principal Component 2)')
    plt.legend()
    plt.savefig(os.path.join(pic_dir, filename))

plot_decision_boundary(X_test_pca, y_test, classifier_viz, 'SVM (RBF Kernel) - PCA Reduced Data', '10-2_Decision_Boundary.png')
