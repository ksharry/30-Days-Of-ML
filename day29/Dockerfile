# 使用官方 Python 3.10 映像檔
FROM python:3.10

# 設定工作目錄
WORKDIR /code

# 複製 requirements.txt 並安裝套件
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 複製所有程式碼
COPY . /code

# 設定 Streamlit 啟動指令
CMD ["streamlit", "run", "Streamlit_App.py", "--server.port", "7860"]
