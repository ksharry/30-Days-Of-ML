# Day06_SVM_Titanic_3D.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 【新增】引入 3D 繪圖工具箱
from mpl_toolkits.mplot3d import Axes3D
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
# 2. 視覺化準備：定義畫 2D 邊界的函數 (維持不變)
# ---------------------------------------------------------
def plot_boundary(model, X, y, ax, title, show_support_vectors=False):
    """
    繪製 2D 決策邊界與資料點的輔助函數
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    if hasattr(model, "decision_function"):
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm_r')
    ax.contour(xx, yy, Z, colors=['k'], levels=[0], linestyles=['-'])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap='coolwarm_r')

    if show_support_vectors and hasattr(model, "support_vectors_"):
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=100, linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')

    ax.set_title(title)
    ax.set_xlabel('Age (Scaled)')
    ax.set_ylabel('Fare (Scaled)')

# ---------------------------------------------------------
# 3. 實驗一：尋找「最寬的馬路」 (Linear Comparison - 2D)
# ---------------------------------------------------------
# 為了 2D 視覺化，只取 Age 和 Fare
X_2d = X[['age', 'fare']].values
y_2d = y.values
scaler_2d = StandardScaler()
X_2d_scaled = scaler_2d.fit_transform(X_2d)

log_reg = LogisticRegression()
log_reg.fit(X_2d_scaled, y_2d)

svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_2d_scaled, y_2d)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plot_boundary(log_reg, X_2d_scaled, y_2d, axes[0], "Logistic Regression (2D)")
plot_boundary(svm_linear, X_2d_scaled, y_2d, axes[1], "SVM Linear (2D - Max Margin)", show_support_vectors=True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4. 實驗二：核函數的魔法 (RBF Kernel - 2D)
# ---------------------------------------------------------
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_2d_scaled, y_2d)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plot_boundary(svm_linear, X_2d_scaled, y_2d, axes[0], "SVM Linear (2D)")
plot_boundary(svm_rbf, X_2d_scaled, y_2d, axes[1], "SVM RBF Kernel (2D Projection)", show_support_vectors=True)
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

# ==============================================================================
# 6. 【新增】額外實驗：三維空間的魔法 (3D Visualization)
# ==============================================================================
print("\nGenerating 3D Plot... Please wait...")

# 6.1 準備 3D 資料 (取 Age, Fare, Pclass)
X_3d = X[['age', 'fare', 'pclass']].values
y_3d = y.values
scaler_3d = StandardScaler()
X_3d_scaled = scaler_3d.fit_transform(X_3d)

# 6.2 訓練 RBF SVM 模型 (在 3D 資料上)
svm_rbf_3d = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf_3d.fit(X_3d_scaled, y_3d)

# 6.3 設定 3D畫布
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# --- 繪製真實資料點 ---
# 紅色=死亡(0), 藍色=生存(1)
scatter = ax.scatter(X_3d_scaled[:, 0], X_3d_scaled[:, 1], X_3d_scaled[:, 2],
                     c=y_3d, cmap='coolwarm_r', s=40, edgecolor='k', label='Actual Data')

# --- 繪製 3D 決策區域 (以半透明散點表示) ---
# 建立 3D 網格
x_min, x_max = X_3d_scaled[:, 0].min() - 0.5, X_3d_scaled[:, 0].max() + 0.5
y_min, y_max = X_3d_scaled[:, 1].min() - 0.5, X_3d_scaled[:, 1].max() + 0.5
z_min, z_max = X_3d_scaled[:, 2].min() - 0.5, X_3d_scaled[:, 2].max() + 0.5
# 降低網格密度以提升繪圖速度 (step=0.2)
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2),
                         np.arange(z_min, z_max, 0.2))

# 預測網格中每個點的類別
Z = svm_rbf_3d.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = Z.reshape(xx.shape)

# 繪製預測區域 (使用低透明度的散點形成「霧氣」效果)
# 這裡只畫出模型預測為「生存 (1)」的區域，用淺藍色表示
ax.scatter(xx[Z==1], yy[Z==1], zz[Z==1], c='blue', alpha=0.03, s=10, marker='s')
# 如果想看死亡區域，取消下面註解：
# ax.scatter(xx[Z==0], yy[Z==0], zz[Z==0], c='red', alpha=0.03, s=10, marker='s')

# 6.4 設定圖表標籤與視角
ax.set_title('SVM (RBF Kernel) 3D Decision Space\nFeatures: Age, Fare, Pclass')
ax.set_xlabel('Age (Scaled)')
ax.set_ylabel('Fare (Scaled)')
ax.set_zlabel('Pclass (Scaled)')

# 調整初始視角 (elev=仰角, azim=方位角)
ax.view_init(elev=20, azim=-60)

plt.legend(*scatter.legend_elements(), title="Survival")
plt.tight_layout()
plt.show()