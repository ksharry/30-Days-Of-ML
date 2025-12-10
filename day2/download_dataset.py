import pandas as pd
from sklearn.datasets import fetch_california_housing

# 載入 California Housing 資料集
print("正在載入 California Housing 資料集...")
housing = fetch_california_housing()

# 建立 DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Target'] = housing.target  # 加入目標變數 (房價)

# 顯示資料集資訊
print(f"\n資料集維度: {df.shape}")
print(f"總共 {df.shape[0]} 筆資料，{df.shape[1]} 個欄位")
print("\n前 5 筆資料:")
print(df.head())

# 儲存為 Excel 檔案
output_file = "california_housing_dataset.xls"
print(f"\n正在儲存資料集為 {output_file}...")

# 使用 xlwt 引擎儲存為 .xls 格式
df.to_excel(output_file, index=False, engine='xlwt')

print(f"✓ 資料集已成功儲存為 {output_file}")
print(f"✓ 檔案位置: c:\\Users\\Harry\\Desktop\\30-Days-Of-ML\\day2\\{output_file}")


