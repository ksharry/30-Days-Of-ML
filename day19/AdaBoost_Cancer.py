# Day 19: AdaBoost (Adaptive Boosting) - 乳癌預測
# ---------------------------------------------------------
# 這一天的目標是學習「Boosting (提升法)」的始祖：AdaBoost。
# 不同於 Random Forest 的「大家一起投票」，AdaBoost 採用「接力賽」的方式。
# 後面的模型會專注於修正前面模型「做錯的題目」。
# 我們使用各個特徵 (如腫瘤大小、質地) 來預測乳癌是良性 (Benign) 還是惡性 (Malignant)。
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 載入乳癌資料集
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target) # 0: Malignant (惡性), 1: Benign (良性)

# 為了方便視覺化決策邊界，我們只取兩個最重要的特徵 (這是在做完 Feature Importance 後選出來的)
# 這裡我們先偷看答案，選 'mean concave points' 和 'worst area'
# (或者我們可以先用全特徵訓練，最後再畫圖。為了教學完整性，我們先用全特徵訓練)
print("Data loaded:")
print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
print(f"Target Names: {data.target_names}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. 設定「弱分類器」 (Weak Learner) ---
# AdaBoost 的核心是把很多「很弱」的模型組合成一個強模型
# 最常用的弱分類器是「決策樹樁 (Decision Stump)」，也就是深度只有 1 的樹
weak_learner = DecisionTreeClassifier(max_depth=1)

# --- 3. 訓練 AdaBoost 模型 ---
# n_estimators=50: 總共接力 50 次
# learning_rate=1.0: 每次修正的力道
# 注意：舊版 sklearn 參數名為 base_estimator，新版為 estimator。這裡為了相容性使用 base_estimator (若報錯請改回 estimator)
try:
    ada_model = AdaBoostClassifier(estimator=weak_learner, n_estimators=50, learning_rate=1.0, random_state=42)
except TypeError:
    ada_model = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=50, learning_rate=1.0, random_state=42)
ada_model.fit(X_train, y_train)

# --- 4. 評估與比較 ---
# 我們也訓練一個單獨的弱分類器來比較
weak_learner.fit(X_train, y_train)

y_pred_ada = ada_model.predict(X_test)
y_pred_weak = weak_learner.predict(X_test)

acc_ada = accuracy_score(y_test, y_pred_ada)
acc_weak = accuracy_score(y_test, y_pred_weak)

print(f"\n--- Accuracy Comparison ---")
print(f"Weak Learner (Stump): {acc_weak:.4f}")
print(f"AdaBoost (50 Stumps): {acc_ada:.4f}")
print(f"Improvement: {(acc_ada - acc_weak)*100:.2f}%")

# --- 5. 視覺化：學習曲線 (Learning Curve) ---
# 看看隨著樹的數量增加，準確率是如何提升的
# AdaBoost 提供 staged_predict 方法，可以取得每一步的預測結果
staged_scores = list(ada_model.staged_score(X_test, y_test))

plt.figure(figsize=(8, 5))
plt.plot(range(1, 51), staged_scores, label='Test Accuracy', color='red')
plt.axhline(y=acc_weak, color='blue', linestyle='--', label='Weak Learner Baseline')
plt.title('AdaBoost Learning Curve')
plt.xlabel('Number of Estimators (Trees)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(pic_dir, '19-1_Learning_Curve.png'))
print("Learning Curve saved.")

# --- 6. 視覺化：決策邊界 (Decision Boundary) ---
# 為了畫出漂亮的 2D 圖，我們只用兩個特徵重新訓練一個簡單的 AdaBoost
feature_cols = ['mean concave points', 'worst area']
X_2d = X[feature_cols]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y, test_size=0.2, random_state=42)

try:
    ada_2d = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
except TypeError:
    ada_2d = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
ada_2d.fit(X_train_2d, y_train_2d)

# 建立網格
x_min, x_max = X_2d.iloc[:, 0].min() - 0.01, X_2d.iloc[:, 0].max() + 0.01
y_min, y_max = X_2d.iloc[:, 1].min() - 10, X_2d.iloc[:, 1].max() + 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.005),
                     np.arange(y_min, y_max, 5))

Z = ada_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
sns.scatterplot(x=X_test_2d.iloc[:, 0], y=X_test_2d.iloc[:, 1], hue=y_test_2d, palette='coolwarm', edgecolor='k')
plt.title('AdaBoost Decision Boundary (2 Features)')
plt.xlabel(feature_cols[0])
plt.ylabel(feature_cols[1])
plt.savefig(os.path.join(pic_dir, '19-2_Decision_Boundary.png'))
print("Decision Boundary saved.")
