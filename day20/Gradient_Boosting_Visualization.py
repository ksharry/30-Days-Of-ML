import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import os

# 設定存檔路徑
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 1. 產生數據 (一個簡單的波浪函數)
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(16)) # 加入一些雜訊

# 2. 訓練 Gradient Boosting (手動模擬)

# 第一棒：擬合原始數據 y
tree1 = DecisionTreeRegressor(max_depth=2)
tree1.fit(X, y)
y1 = tree1.predict(X)

# 計算殘差 (Residual)
y2_residual = y - y1

# 第二棒：擬合殘差 y2_residual
tree2 = DecisionTreeRegressor(max_depth=2)
tree2.fit(X, y2_residual)
y2 = tree2.predict(X)

# 第三棒：擬合剩下的殘差
y3_residual = y2_residual - y2
tree3 = DecisionTreeRegressor(max_depth=2)
tree3.fit(X, y3_residual)
y3 = tree3.predict(X)

# 組合預測
y_pred_1 = y1
y_pred_2 = y1 + y2
y_pred_3 = y1 + y2 + y3

# 3. 繪圖
plt.figure(figsize=(12, 10))

# 子圖 1: 第一棒的表現
plt.subplot(3, 2, 1)
plt.scatter(X, y, color='black', s=10, label='Data')
plt.plot(X, y_pred_1, color='red', linewidth=2, label='Tree 1 (Model 1)')
plt.title('Iteration 1: Fit Original Data')
plt.legend()

# 子圖 2: 第一棒的殘差 (第二棒的目標)
plt.subplot(3, 2, 2)
plt.scatter(X, y2_residual, color='green', s=10, label='Residual 1')
plt.plot(X, y2, color='blue', linewidth=2, label='Tree 2 (Fit Residual)')
plt.title('Iteration 2: Fit Residual of Tree 1')
plt.legend()

# 子圖 3: 第一棒 + 第二棒
plt.subplot(3, 2, 3)
plt.scatter(X, y, color='black', s=10, label='Data')
plt.plot(X, y_pred_2, color='red', linewidth=2, label='Tree 1 + Tree 2')
plt.title('Combined: Tree 1 + Tree 2')
plt.legend()

# 子圖 4: 第二棒的殘差 (第三棒的目標)
plt.subplot(3, 2, 4)
plt.scatter(X, y3_residual, color='green', s=10, label='Residual 2')
plt.plot(X, y3, color='blue', linewidth=2, label='Tree 3 (Fit Residual)')
plt.title('Iteration 3: Fit Residual of Tree 2')
plt.legend()

# 子圖 5: 全部加總 (Tree 1 + 2 + 3)
plt.subplot(3, 2, 5)
plt.scatter(X, y, color='black', s=10, label='Data')
plt.plot(X, y_pred_3, color='red', linewidth=2, label='Tree 1 + Tree 2 + Tree 3')
plt.title('Final: Tree 1 + Tree 2 + Tree 3')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '20-4_Gradient_Boosting_Process.png'))
print("Gradient Boosting Process plot saved.")
