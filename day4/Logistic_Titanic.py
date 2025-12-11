# Day04_Logistic_Titanic.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ---------------------------------------------------------
# 1. 載入與清洗數據 (Data Preparation)
# ---------------------------------------------------------
# 使用 Seaborn 內建的鐵達尼號資料集
df = sns.load_dataset('titanic')

# 簡單的預處理：
# - 填補 Age 的缺失值 (用中位數)
# - 丟棄缺失過多的 'deck' 欄位
# - 填補 Embarked 缺失值
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# 選取特徵 (Feature Selection)
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = df[features]
y = df['survived']

# 獨熱編碼 (One-Hot Encoding): 將文字類別 (sex, embarked) 轉為數字
X = pd.get_dummies(X, columns=['sex', 'embarked'], drop_first=True)

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化 (Standardization): 邏輯回歸對數值尺度很敏感，務必縮放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 2. 訓練模型 (Model Training)
# ---------------------------------------------------------
# 建立邏輯回歸模型
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 進行預測
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1] # 取得預測為 "1 (生存)" 的機率

# ---------------------------------------------------------
# 3. 評估與視覺化 (Evaluation & Visualization)
# ---------------------------------------------------------
# 計算分數
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 設定畫布
plt.figure(figsize=(12, 5))

# 圖一：混淆矩陣 (Confusion Matrix)
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['Dead (0)', 'Survived (1)'])
plt.yticks([0.5, 1.5], ['Dead (0)', 'Survived (1)'])

# 圖二：特徵重要性 (Feature Coefficients)
# 邏輯回歸的係數代表該特徵對 "Log-Odds" 的影響力
plt.subplot(1, 2, 2)
coefs = pd.Series(model.coef_[0], index=X.columns)
coefs.sort_values().plot(kind='barh', color=np.where(coefs.sort_values() > 0, 'green', 'red'))
plt.title('Feature Importance (Coefficients)')
plt.xlabel('Coefficient Value (Impact on Survival Odds)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.show()

# 額外：印出 Sigmoid 函數概念
print("\n[Concept Check]")
print(f"係數解讀: Sex_male 的權重是 {coefs['sex_male']:.2f}。")
print("這表示男性會大幅 '降低' 生存機率 (負值)。")