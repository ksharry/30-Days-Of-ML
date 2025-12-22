# Day 07: K-近鄰演算法 (K-Nearest Neighbors) - 鳶尾花分類
# ---------------------------------------------------------
# 這一天的目標是學習最直覺的分類演算法：KNN。
# 我們使用經典的 Iris (鳶尾花) 資料集。
# 重點：歐式距離、K 值選擇、多類別分類。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import ListedColormap

# 模型與評估工具
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# --- 1. 載入資料 (Data Loading) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 載入 Iris 資料集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 轉為 DataFrame 方便觀察
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['species'] = df['target'].map({0: target_names[0], 1: target_names[1], 2: target_names[2]})

print(f"資料集維度: {df.shape}")
print(df.head())

# --- 2. 數據觀察 (EDA) ---
# 使用 Pairplot 觀察特徵之間的關係
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue='species', palette='Set1')
plt.savefig(os.path.join(pic_dir, '7-1_EDA_Pairplot.png'))
# plt.show()

# --- 3. 資料分割與前處理 ---
# 為了方便視覺化決策邊界，我們先只選取兩個特徵進行訓練 (例如：sepal length 和 sepal width)
# 但為了模型效能，通常會用所有特徵。這裡我們先用所有特徵訓練一次，後面再用兩個特徵畫圖。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 標準化 (Standardization) - 對 KNN 非常重要！
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. 建立與訓練模型 ---
# 選擇 K=5
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train_scaled, y_train)

# --- 5. 模型評估 ---
y_pred = classifier.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

metrics_output = f"""
Accuracy: {acc:.4f}
Confusion Matrix:
{cm}

Classification Report:
{classification_report(y_test, y_pred, target_names=target_names)}
"""

print(metrics_output)
with open(os.path.join(SCRIPT_DIR, 'metrics.txt'), 'w') as f:
    f.write(metrics_output)

# 繪製混淆矩陣
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix (KNN, K=5)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(pic_dir, '7-2_Confusion_Matrix.png'))

# --- 6. 結果視覺化 (決策邊界 Decision Boundary) ---
# 為了畫出 2D 邊界，我們必須重新訓練一個只用兩個特徵的模型
# 我們選用 'sepal length (cm)' (index 0) 和 'petal length (cm)' (index 2) 
# 因為從 pairplot 看起來這兩個分得比較開 (或者選 petal length & petal width)
# 讓我們選 petal length (2) 和 petal width (3) 效果通常最好
feature_idx = [2, 3] 
X_2d = X[:, feature_idx]
y_2d = y

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y_2d, test_size=0.25, random_state=0)

scaler_2d = StandardScaler()
X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
X_test_2d_scaled = scaler_2d.transform(X_test_2d)

classifier_2d = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier_2d.fit(X_train_2d_scaled, y_train_2d)

def plot_decision_boundary(X_set, y_set, model, title, filename):
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    
    plt.figure(figsize=(10, 6))
    # 預測網格中每個點的類別
    Z = model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    
    plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green', 'blue'))(i), label=target_names[j], edgecolor='black')
        
    plt.title(title)
    plt.xlabel(f'{feature_names[feature_idx[0]]} (Scaled)')
    plt.ylabel(f'{feature_names[feature_idx[1]]} (Scaled)')
    plt.legend()
    plt.savefig(os.path.join(pic_dir, filename))

plot_decision_boundary(X_test_2d_scaled, y_test_2d, classifier_2d, 'KNN (K=5) - Petal Length vs Width', '7-3_Decision_Boundary.png')

# --- 額外：尋找最佳 K 值 ---
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_scaled, y_train)
    pred_i = knn.predict(X_test_scaled)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.savefig(os.path.join(pic_dir, '7-4_K_Value_Error.png'))
