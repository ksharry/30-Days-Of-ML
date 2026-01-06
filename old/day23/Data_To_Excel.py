# Data_To_Excel.py
import pandas as pd
import requests
import zipfile
import io
import os

def download_and_convert_to_excel():
    print("1. 正在從 UCI Repository 下載資料...")
    # UCI SMS Spam Collection 的下載連結
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # 檢查請求是否成功
        
        # 在記憶體中解壓縮
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # UCI 的 zip 裡面有一個檔案叫做 'SMSSpamCollection'
            if 'SMSSpamCollection' in z.namelist():
                print("2. 下載成功！正在讀取檔案...")
                with z.open('SMSSpamCollection') as f:
                    # 讀取 tab 分隔的檔案
                    df = pd.read_csv(f, sep='\t', header=None, names=['label', 'text'])
                
                print(f"   資料讀取完畢，共有 {len(df)} 筆簡訊。")
                print("   前 5 筆資料預覽：")
                print(df.head())
                
                # 存成 Excel
                output_filename = "UCI_Spam_Data.xlsx"
                print(f"3. 正在寫入 Excel ({output_filename})...")
                df.to_excel(output_filename, index=False)
                print(f"★ 完成！請開啟 {output_filename} 查看資料。")
                
            else:
                print("錯誤：壓縮檔中找不到 SMSSpamCollection 檔案。")
                
    except Exception as e:
        print(f"發生錯誤：{e}")

if __name__ == "__main__":
    download_and_convert_to_excel()