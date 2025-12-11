# Day06_SVM_Titanic.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. 資料準備 (標準流程)
# ---------------------------------------------------------
df = sns.load_dataset('titanic')
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = df[features]
y = df['survived']
X = pd.get_dummies(X, columns=['sex', 'embarked'], drop_first=True)

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ★★★ 關鍵：SVM 對縮放極度敏感，一定要標準化 ★★★
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 2. 視覺化準備：定義畫邊界的函數
# ---------------------------------------------------------
def plot_boundary(model, X, y, ax, title, show_support_vectors=False):
    """
    繪製 2D 決策邊界與資料點的輔助函數
    """
    # 建立網格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    # 預測網格分數 (使用 decision_function 畫出漸層背景)
    if hasattr(model, "decision_function"):
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    Z = Z.reshape(xx.shape)

    # 繪製等高線背景 (紅藍配色：紅死藍生)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm_r')
    # 繪製決策邊界線 (分數為 0 的地方)
    ax.contour(xx, yy, Z, colors=['k'], levels=[0], linestyles=['-'])

    # 繪製資料點
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap='coolwarm_r')

    # 標記支援向量 (Support Vectors)
    if show_support_vectors and hasattr(model, "support_vectors_"):
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=100, linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')

    ax.set_title(title)
    ax.set_xlabel('Age (Scaled)')
    ax.set_ylabel('Fare (Scaled)')

# ---------------------------------------------------------
# 3. 實驗一：尋找「最寬的馬路」 (Linear Comparison)
# ---------------------------------------------------------
# 為了視覺化，只取 Age 和 Fare
X_2d = X[['age', 'fare']].values
y_2d = y.values
scaler_2d = StandardScaler()
X_2d_scaled = scaler_2d.fit_transform(X_2d)

# 訓練模型
log_reg = LogisticRegression()
log_reg.fit(X_2d_scaled, y_2d)

svm_linear = SVC(kernel='linear', C=1.0) # 線性核函數
svm_linear.fit(X_2d_scaled, y_2d)

# 繪圖
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plot_boundary(log_reg, X_2d_scaled, y_2d, axes[0], "Logistic Regression (Any Line)")
# SVM 開啟支援向量標記
plot_boundary(svm_linear, X_2d_scaled, y_2d, axes[1], "SVM Linear (Widest Road)", show_support_vectors=True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4. 實驗二：核函數的魔法 (Kernel Trick)
# ---------------------------------------------------------
# 訓練非線性 SVM (RBF Kernel)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale') # 徑向基核函數
svm_rbf.fit(X_2d_scaled, y_2d)

# 繪圖
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plot_boundary(svm_linear, X_2d_scaled, y_2d, axes[0], "SVM Linear (Flat World)", show_support_vectors=True)
plot_boundary(svm_rbf, X_2d_scaled, y_2d, axes[1], "SVM RBF Kernel (High-Dim Magic)", show_support_vectors=True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5. 最終效能評估 (使用完整特徵)
# ---------------------------------------------------------
final_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
final_svm.fit(X_train_scaled, y_train)
y_pred = final_svm.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nFinal SVM (RBF) Accuracy on full features: {acc:.4f}")