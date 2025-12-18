# Day06_SVM_Tuning.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 引入 GridSearch 用於自動調參
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# 引入 3D 繪圖
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------
# 1. 資料準備 (Data Preparation)
# ---------------------------------------------------------
print("1. 正在處理資料...")
df = sns.load_dataset('titanic')
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = df[features]
y = df['survived']
X = pd.get_dummies(X, columns=['sex', 'embarked'], drop_first=True)

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ★★★ 標準化 (Standardization) ★★★
# SVM 對距離極度敏感，務必縮放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 2. 自動調參 (Grid Search Cross Validation)
# ---------------------------------------------------------
print("2. 啟動 Grid Search 尋找最佳參數 (這可能需要幾秒鐘)...")

# 設定參數候選範圍
# C: 懲罰力度 (越小越佛系，越大越嚴格)
# gamma: 影響範圍 (越小越廣，越大越窄)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# 建立 Grid Search
# refit=True 代表找到最佳參數後，會自動用該參數重新訓練一次模型
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
grid.fit(X_train_scaled, y_train)

# 取得最佳參數與模型
best_params = grid.best_params_
best_model = grid.best_estimator_

print("-" * 40)
print(f"★ 最佳參數組合: {best_params}")
print(f"★ 最佳訓練分數: {grid.best_score_:.4f}")
print("-" * 40)

# 用最佳參數預測測試集
y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"最終測試集準確率 (Test Accuracy): {acc:.4f}")

# ---------------------------------------------------------
# 3. 視覺化對比：預設 vs 最佳 (2D Projection)
# ---------------------------------------------------------
print("3. 繪製 2D 決策邊界對比圖...")

# 為了視覺化，我們需要重新訓練只包含 Age 和 Fare 的 2D 模型
X_2d = X[['age', 'fare']].values
y_2d = y.values
scaler_2d = StandardScaler()
X_2d_scaled = scaler_2d.fit_transform(X_2d)

# A. 預設模型 (C=1, gamma='scale')
model_default = SVC(kernel='rbf', C=1.0, gamma='scale')
model_default.fit(X_2d_scaled, y_2d)

# B. 最佳模型 (使用 Grid Search 找到的 C 和 gamma)
model_best = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
model_best.fit(X_2d_scaled, y_2d)

# 定義繪圖函數
def plot_boundary(model, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm_r')
    ax.contour(xx, yy, Z, colors=['k'], levels=[0], linestyles=['-'])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap='coolwarm_r')
    ax.set_title(title)
    ax.set_xlabel('Age (Scaled)')
    ax.set_ylabel('Fare (Scaled)')

# 繪製對比圖
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plot_boundary(model_default, X_2d_scaled, y_2d, axes[0], "Default Parameters (C=1, gamma='scale')")
plot_boundary(model_best, X_2d_scaled, y_2d, axes[1], f"Tuned Best Parameters (C={best_params['C']}, g={best_params['gamma']})")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4. 3D 視覺化：使用最佳參數 (The Grand Finale)
# ---------------------------------------------------------
print("4. 繪製 3D 最佳決策空間...")

# 準備 3D 資料
X_3d = X[['age', 'fare', 'pclass']].values
scaler_3d = StandardScaler()
X_3d_scaled = scaler_3d.fit_transform(X_3d)

# 使用最佳參數訓練 3D 模型
model_3d = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
model_3d.fit(X_3d_scaled, y.values)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 繪製真實數據點
scatter = ax.scatter(X_3d_scaled[:, 0], X_3d_scaled[:, 1], X_3d_scaled[:, 2],
                     c=y, cmap='coolwarm_r', s=40, edgecolor='k', label='Actual Data')

# 繪製決策區域 (霧氣)
x_min, x_max = X_3d_scaled[:, 0].min() - 0.5, X_3d_scaled[:, 0].max() + 0.5
y_min, y_max = X_3d_scaled[:, 1].min() - 0.5, X_3d_scaled[:, 1].max() + 0.5
z_min, z_max = X_3d_scaled[:, 2].min() - 0.5, X_3d_scaled[:, 2].max() + 0.5

xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2),
                         np.arange(z_min, z_max, 0.2))

Z = model_3d.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = Z.reshape(xx.shape)

# 只畫出生存區 (1)
ax.scatter(xx[Z==1], yy[Z==1], zz[Z==1], c='blue', alpha=0.03, s=10, marker='s')

ax.set_title(f'SVM 3D Decision Space (Tuned: C={best_params["C"]})\nFeatures: Age, Fare, Pclass')
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.set_zlabel('Pclass')
ax.view_init(elev=20, azim=-60)
plt.legend(*scatter.legend_elements(), title="Survival")
plt.show()

print("程式執行完畢！")