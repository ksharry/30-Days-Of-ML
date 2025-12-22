import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 模型與評估工具
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- 1. 載入資料 (Data Loading) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'titanic.csv')

if not os.path.exists(DATA_FILE):
    print(f"錯誤：找不到檔案 {DATA_FILE}")
    print("請手動下載資料集或檢查路徑。")
    exit()

df = pd.read_csv(DATA_FILE)

# --- 2. 資料清理與前處理 (Preprocessing) ---
# 為了簡化，我們只選取幾個關鍵特徵
# Pclass: 艙等 (1, 2, 3)
# Sex: 性別 (male, female)
# Age: 年齡
# SibSp: 兄弟姊妹/配偶數
# Parch: 父母/小孩數
# Fare: 票價
# Embarked: 登船港口 (C, Q, S)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# 填補缺失值
# Age 用中位數填補
df['Age'].fillna(df['Age'].median(), inplace=True)
# Embarked 用眾數填補
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# Fare 用中位數填補 (雖然通常只有 test set 會有缺，但保險起見)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# 類別特徵編碼 (Label Encoding)
# 決策樹其實可以處理類別特徵，但 sklearn 的實作目前只支援數值輸入
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex']) # male:1, female:0

le_embarked = LabelEncoder()
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

X = df[features]
y = df[target]

print(f"資料集維度: {df.shape}")
print(X.head())

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# --- 3. 建立與訓練模型 ---
# criterion='entropy': 使用資訊增益 (Information Gain) 來分裂
# max_depth=3: 限制樹的深度，避免過擬合 (Overfitting)，也方便視覺化
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
classifier.fit(X_train, y_train)

# --- 4. 模型評估 ---
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

metrics_output = f"""
Accuracy: {acc:.4f}
Confusion Matrix:
{cm}

Classification Report:
{classification_report(y_test, y_pred)}
"""

print(metrics_output)
with open(os.path.join(SCRIPT_DIR, 'metrics.txt'), 'w') as f:
    f.write(metrics_output)

pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# 繪製混淆矩陣
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Decision Tree)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(pic_dir, '9-1_Confusion_Matrix.png'))

# --- 5. 結果視覺化 (決策樹圖) ---
# 這是 Day 09 的重頭戲！畫出決策樹的樣子
plt.figure(figsize=(20, 10))
plot_tree(classifier, 
          feature_names=features, 
          class_names=['Not Survived', 'Survived'], 
          filled=True, 
          rounded=True, 
          fontsize=12)
plt.title('Decision Tree Visualization (Depth=3)')
plt.savefig(os.path.join(pic_dir, '9-2_Decision_Tree.png'))

# --- 額外：特徵重要性 (Feature Importance) ---
# 看看決策樹認為哪個特徵最重要
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center", color='orange')
plt.xticks(range(X.shape[1]), [features[i] for i in indices])
plt.xlim([-1, X.shape[1]])
plt.savefig(os.path.join(pic_dir, '9-3_Feature_Importance.png'))
