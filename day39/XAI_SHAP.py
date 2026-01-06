import xgboost
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# 建立結果資料夾
os.makedirs("day39/pic", exist_ok=True)

# === 1. 準備資料 ===
# 使用 SHAP 內建的 California Housing 資料集 (取代舊的 Boston Housing)
X, y = shap.datasets.california()

# 為了示範清楚，我們只取前 1000 筆資料來跑
X100 = shap.utils.sample(X, 100) 

print("資料載入完成。特徵包含：")
print(X.columns)

# === 2. 訓練模型 (XGBoost) ===
print("正在訓練 XGBoost 模型...")
model = xgboost.XGBRegressor().fit(X, y)

# === 3. 解釋模型 (SHAP) ===
print("正在計算 SHAP 值 (這可能需要一點時間)...")

# TreeExplainer 專門用來解釋樹模型 (XGBoost, Random Forest 等)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# === 4. 畫圖 ===

# (1) Summary Plot: 全局解釋
# 展示每個特徵的重要性，以及它如何影響預測 (紅色代表數值高，藍色代表數值低)
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Summary Plot (Feature Importance)")
plt.tight_layout()
plt.savefig('day39/pic/shap_summary.png')
print("已儲存全局解釋圖: day39/pic/shap_summary.png")
plt.close()

# (2) Force Plot: 局部解釋 (解釋第一筆資料)
# 為什麼 AI 認為「這間房子」值這個錢？
# 注意: force_plot 本身是 JS 互動圖，無法直接存成 png。
# 這裡我們改用 waterfall plot (瀑布圖) 來解釋單筆資料，這在靜態圖片上更清楚。

plt.figure()
# 創建一個 Explanation 物件 (新版 SHAP 需要)
shap_explanation = shap.Explanation(values=shap_values[0], 
                                    base_values=explainer.expected_value, 
                                    data=X.iloc[0], 
                                    feature_names=X.columns)

shap.plots.waterfall(shap_explanation, show=False)
plt.title("SHAP Waterfall Plot (Single Prediction)")
plt.tight_layout()
plt.savefig('day39/pic/shap_waterfall.png')
print("已儲存單筆解釋圖: day39/pic/shap_waterfall.png")
plt.close()

print("\n完成！請查看 pic 資料夾中的圖片。")
print("1. shap_summary.png: 告訴你哪些特徵最重要 (例如 Latitude, Longitude, MedInc)")
print("2. shap_waterfall.png: 告訴你為什麼第一間房子的預測價是這樣 (從平均值開始加減)")
