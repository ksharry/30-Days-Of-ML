import seaborn as sns
import pandas as pd
import os

# 1. 載入資料 (Load Data)
print("正在載入鐵達尼號資料集 (Loading Titanic dataset)...")
df = sns.load_dataset('titanic')

# 2. 設定輸出檔名 (Set Output Filename)
output_file = 'titanic_data.xlsx'

# 3. 儲存為 Excel (Save to Excel)
print(f"正在將資料寫入 {output_file} (Saving data to {output_file})...")
try:
    df.to_excel(output_file, index=False)
    print(f"成功！資料已儲存至 {os.path.abspath(output_file)}")
except Exception as e:
    print(f"發生錯誤 (Error): {e}")
