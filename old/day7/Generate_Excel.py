import pandas as pd
from sklearn.datasets import fetch_openml
import os

def download_credit_data():
    print("⏳ 正在從 OpenML 下載 German Credit Data，請稍候...")
    
    try:
        # 1. 抓取資料 (version=1 是最通用的版本)
        # as_frame=True 會直接回傳 pandas DataFrame 格式
        credit_data = fetch_openml(name='credit-g', version=1, as_frame=True)
        
        # 2. 取得資料表內容
        df = credit_data.frame
        
        # 3. 簡單檢視一下
        print(f"✅ 下載成功！資料大小：{df.shape[0]} 筆資料, {df.shape[1]} 個欄位")
        
        # 4. 定義檔案名稱
        filename = "german_credit_data.csv"
        
        # 5. 存成 CSV 檔案 (index=False 代表不存入 0,1,2... 這種索引行)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        # 6. 告訴使用者檔案在哪裡
        current_path = os.getcwd()
        full_path = os.path.join(current_path, filename)
        
        print("-" * 30)
        print(f"🎉 檔案已儲存！")
        print(f"📂 檔案位置: {full_path}")
        print("-" * 30)
        print("您現在可以使用 Excel 開啟這個檔案，或在 Python 中使用 pd.read_csv() 讀取它。")
        
    except Exception as e:
        print("❌ 下載失敗，錯誤訊息如下：")
        print(e)

if __name__ == "__main__":
    download_credit_data()