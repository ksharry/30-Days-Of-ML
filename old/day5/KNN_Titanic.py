# Day05_KNN_Titanic.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------
# 1. 資料準備 (Data Preparation)
# ---------------------------------------------------------
df = sns.load_dataset('titanic')

# 填補缺失值
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# 選取特徵
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = df[features]
y = df['survived']

# 獨熱編碼
X = pd.get_dummies(X, columns=['sex', 'embarked'], drop_first=True)

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ★★★ 關鍵步驟：標準化 (Standardization) ★★★
# KNN 是基於「距離」的算法。Fare (0-500) 會主導 Age (0-80) 的距離計算。
# 務必將所有特徵縮放到同一尺度 (Mean=0, Std=1)。
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 2. 圖一：決策邊界比較 (Decision Boundary Comparison)
# ---------------------------------------------------------
# 為了畫出 2D 平面圖，我們只取 'Age' 和 'Fare' 兩個特徵重新訓練
X_2d_train = X_train[['age', 'fare']].values
X_2d_test = X_test[['age', 'fare']].values

# 針對 2D 資料單獨做標準化
scaler_2d = StandardScaler()
X_2d_train_scaled = scaler_2d.fit_transform(X_2d_train)

# 訓練兩個模型
clf_log = LogisticRegression().fit(X_2d_train_scaled, y_train)
# 這裡設定 K=17 (參考您提供的最佳值)
clf_knn = KNeighborsClassifier(n_neighbors=17).fit(X_2d_train_scaled, y_train)

# 建立網格 (Meshgrid)
x_min, x_max = X_2d_train_scaled[:, 0].min() - 0.5, X_2d_train_scaled[:, 0].max() + 0.5
y_min, y_max = X_2d_train_scaled[:, 1].min() - 0.5, X_2d_train_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# 進行網格預測
Z_log = clf_log.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_knn = clf_knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 開始繪圖 - 圖一
plt.figure(figsize=(16, 8))

# 左圖：邏輯回歸
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_log, alpha=0.4, cmap='Blues')
plt.scatter(X_2d_train_scaled[:, 0], X_2d_train_scaled[:, 1], c=y_train, s=30, edgecolor='k', cmap='Blues')
plt.title('Logistic Regression Decision Boundary\n(Linear Line)', fontsize=14)
plt.xlabel('Age (Scaled)')
plt.ylabel('Fare (Scaled)')

# 右圖：KNN
plt.subplot(1, 2, 2)
# 使用自定義顏色讓邊界更明顯 (紫色系)
cmap_light = ListedColormap(['#CAB2D6', '#6A3D9A']) # 淺紫, 深紫
plt.contourf(xx, yy, Z_knn, alpha=0.4, cmap='Purples')
plt.scatter(X_2d_train_scaled[:, 0], X_2d_train_scaled[:, 1], c=y_train, s=30, edgecolor='k', cmap='Purples')
plt.title(f'KNN (K=17) Decision Boundary\n(Non-Linear / Organic)', fontsize=14)
plt.xlabel('Age (Scaled)')
plt.ylabel('Fare (Scaled)')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 3. 圖二：K 值尋優與混淆矩陣 (K-Optimization & Confusion Matrix)
# ---------------------------------------------------------
# 回到使用「所有特徵」的完整資料集
k_range = range(1, 41)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores = knn.score(X_test_scaled, y_test)
    k_scores.append(scores)

# 找出分數最高的 K
best_k = k_range[np.argmax(k_scores)]
best_score = max(k_scores)

# 使用最佳 K 進行最終預測
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred_knn = knn_best.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)

# 開始繪圖 - 圖二
plt.figure(figsize=(16, 7))

# 左圖：Accuracy vs K
plt.subplot(1, 2, 1)
plt.plot(k_range, k_scores, marker='o', linestyle='-', color='purple', linewidth=2)
plt.axvline(best_k, color='r', linestyle='--', label=f'Best K={best_k}')
plt.title('Accuracy vs. K Value', fontsize=14)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 右圖：Confusion Matrix
plt.subplot(1, 2, 2)
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Purples', cbar=False, annot_kws={"size": 12})
plt.title(f'KNN Confusion Matrix (K={best_k})\nAccuracy: {acc_knn:.2%}', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['Dead', 'Survived'])
plt.yticks([0.5, 1.5], ['Dead', 'Survived'])

plt.tight_layout()
plt.show()

print(f"最佳 K 值: {best_k}")
print(f"KNN 模型準確率: {acc_knn:.4f}")