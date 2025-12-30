# Day 29: 模型部署 (Model Deployment) - Streamlit Web App

## 0. 歷史小故事/核心貢獻者:
在過去，要將 AI 模型變成一個網頁應用，你需要學會 HTML, CSS, JavaScript, Flask/Django 等一堆繁瑣的技術。
**Streamlit** 的出現改變了一切。它讓資料科學家可以用 **純 Python** 快速打造出漂亮的 Web App，完全不需要懂前端技術。
現在，它是 AI 領域最流行的展示工具 (Demo Tool)。

## 1. 專案目標
### 目標：打造一個 AI 影像辨識 App
我們將使用 **MobileNetV2** (一個輕量級但強大的預訓練模型)，把它包裝成一個網頁。
使用者可以：
1.  上傳圖片。
2.  調整信心門檻。
3.  看到 AI 的辨識結果與信心條。

## 2. 原理

### 2.1 為什麼選擇 MobileNetV2？
在 Web App 中，**速度 (Speed)** 往往比極致的準確度更重要。
*   **巨型模型 (如 ResNet152)**：準確度高，但檔案大 (數百 MB)，推論慢 (CPU 可能要跑好幾秒)。
*   **輕量級模型 (如 MobileNetV2)**：
    *   **體積小**：只有約 14 MB。
    *   **速度快**：專為手機/網頁設計，CPU 也能秒殺。
    *   **準度夠**：在 ImageNet 上仍有很好的表現。
    *   **能辨識什麼？**：它經過 **ImageNet** 資料集訓練，能辨識 **1000 種** 常見物體，包括：
        *   **動物**：各種品種的貓、狗、鳥、魚、昆蟲...
        *   **交通工具**：跑車、卡車、飛機、腳踏車...
        *   **日常用品**：水瓶、鍵盤、滑鼠、雨傘...
        *   **食物**：漢堡、披薩、冰淇淋...

### 2.2 Streamlit 架構：
![Streamlit Architecture](pic/29-2_Streamlit_Architecture.png)

這就是 Streamlit 最神奇的地方！它採用了 **「腳本式執行 (Script Execution)」** 模式：

1.  **你寫 Python (Backend)**：你只需要寫 `st.button()` 或 `st.image()`。
2.  **Streamlit 翻譯 (Middle)**：Streamlit Server 會把你的 Python 指令翻譯成前端看得懂的訊號 (JSON)。
3.  **瀏覽器渲染 (Frontend)**：瀏覽器收到訊號後，自動畫出漂亮的按鈕和圖片 (這些是預先寫好的 React 元件)。

**流程總結**：
*   **互動即重跑**：當使用者按按鈕，Streamlit 會 **從頭到尾重新執行一次** 你的 Python 腳本。
*   **快取魔法**：為了不讓模型每次都重載，我們使用 `@st.cache_resource` 來記住模型。

### 2.3 模型從哪來？為什麼我沒看到檔案？
你可能會疑惑：「我沒有下載 `.h5` 模型檔，為什麼能跑？」
*   **自動下載 (Auto-download)**：當程式執行到 `MobileNetV2(weights='imagenet')` 時，Keras 會自動去 Google 的伺服器下載訓練好的權重 (約 14MB)。
*   **快取位置 (Cache)**：下載後，檔案會被藏在你的電腦裡 (通常在 `C:\Users\Harry\.keras\models`)。
*   **下次秒開**：第二次執行時，程式會直接讀取這個快取檔，不用再下載。
這就是使用 **Keras Applications** (預訓練模型庫) 的方便之處！

### 2.4 為什麼需要「信心門檻」？
在側邊欄，我們設計了一個 **Slider (滑桿)** 來調整信心門檻，這有什麼用？
*   **AI 的輸出是機率**：模型不會只給一個答案，而是給出 1000 個類別的機率分佈。
*   **過濾雜訊**：有時候 AI 對某張圖很不確定 (例如最高分只有 10%)。這時候如果硬要顯示結果，可能會誤導使用者。
*   **機制**：透過設定門檻 (例如 20%，即機率 **0.2**)，我們可以告訴 AI：「如果你沒有 20% 的把握，就別亂猜。」
    *   *註：在 1000 類別中，隨機亂猜的機率只有 0.1% (0.001)。所以 20% 其實已經是相當高的信心了！*

## 3. 實戰
### 3.1 安裝 Streamlit
在執行之前，你需要安裝 `streamlit` 套件：
```bash
pip install streamlit
```

### 3.2 Python 程式碼實作
完整程式連結：[Streamlit_App.py](Streamlit_App.py)

```python
# 關鍵程式碼：顯示圖片與進度條
import streamlit as st

st.title("AI 影像辨識 App")
uploaded_file = st.file_uploader("上傳圖片...")

if uploaded_file:
    st.image(uploaded_file)
    # ... 預測代碼 ...
    st.write(f"結果: {label}")
    st.progress(score)
```

### 3.3 如何執行 App？
**注意！** Streamlit App 不能直接用 `python script.py` 執行。
請在終端機 (Terminal) 輸入以下指令：

```bash
streamlit run day29/Streamlit_App.py
```

執行後，瀏覽器會自動打開一個分頁 (通常是 `http://localhost:8501`)，你就可以看到你的 App 了！

## 4. 成果展示
### 4.1 預期效果
1.  **介面**：你會看到一個簡潔的標題和上傳區塊。
2.  **互動**：上傳一張貓的照片，AI 會告訴你它是 "tabby (虎斑貓)" 或 "Egyptian_cat (埃及貓)"，並顯示信心分數。
3.  **側邊欄**：你可以滑動側邊欄的 Slider 來過濾掉信心度太低的預測。

### 4.2 結果分析：83%的判斷率？
![Result Analysis](pic/29-1.jpg)
*(上圖為使用者實測結果)*

你可能會覺得 **83.43%** 的信心度好像不夠高，但在 ImageNet (1000 類別) 的挑戰中，這其實是非常好的表現！
1.  **千中選一**：模型必須從 1000 種物體中選出正確的那一個，隨機猜對率只有 0.1%。
2.  **相似物種干擾**：黃金獵犬 (Golden Retriever) 和 拉布拉多 (Labrador) 小時候長得很像。模型能有 83% 的把握，代表它捕捉到了關鍵特徵 (如毛髮長度)。
3.  **輕量級模型**：我們使用的是 **MobileNetV2**，它是為了手機/網頁設計的「輕量級」模型。雖然準確度略低於巨型模型 (如 ResNet152)，但它的**速度快非常多**，非常適合 Web App。

### 4.3 失敗案例分析：為什麼認不出蘋果？
![Apple Fail](pic/29-3_Apple_Fail.jpg)
*(上圖：AI 把蘋果誤判為南瓜或石榴)*

你可能會發現，上傳一顆又紅又大的蘋果，AI 竟然認不出來！
*   **原因**：ImageNet 的 1000 個類別中，**其實沒有「紅蘋果 (Red Apple)」這個類別**！(只有 `Granny_Smith` 青蘋果)。
*   **AI 的限制**：AI 只能辨識它**學過**的東西。如果類別表裡沒有蘋果，它就會試圖從它學過的東西裡找最像的 (例如：石榴、無花果、網球)。
*   **啟示**：這告訴我們，**預訓練模型不是萬能的**。如果你需要辨識特定產品 (如自家工廠的零件)，你必須使用 **Transfer Learning (遷移學習)** (如 Day 26) 來重新訓練模型。

## 5. 戰略總結: 從 Lab 到 Production

| 階段 | 工具/環境 | 重點 |
| :--- | :--- | :--- |
| **實驗 (Lab)** | Jupyter Notebook | 快速試錯、視覺化、數據分析。 |
| **開發 (Dev)** | VS Code + Python Scripts | 模組化、重構程式碼、版本控制 (Git)。 |
| **部署 (Prod)** | **Streamlit** / Flask / FastAPI | **使用者介面 (UI)**、API 服務、讓非技術人員也能使用。 |

####  如果未來想要正式推出？

### 6.1 如何擁有自己的網址？(Streamlit Cloud)
如果你想把這個 App 傳給朋友用，最簡單的方法是使用 **Streamlit Cloud** (完全免費)：
1.  **上傳 GitHub**：把你的程式碼 (`Streamlit_App.py` 和 `requirements.txt`) 上傳到 GitHub。
2.  **連結帳號**：去 [share.streamlit.io](https://share.streamlit.io/) 註冊並連結你的 GitHub。
3.  **一鍵部署**：選擇你的 Repository，點擊 "Deploy"。
4.  **獲得網址**：幾分鐘後，你就會獲得一個 `https://your-app.streamlit.app` 的網址，全世界都能訪問！

### 6.2 邁向百萬用戶 (Production)
Streamlit 非常適合做 Demo 或內部工具。但如果你想打造一個**百萬人使用的正式產品**，建議採取以下步驟：

1.  **容器化 (Docker)**：
    *   把你的 App 和環境打包成一個 Docker Image。這樣無論搬到哪台伺服器 (AWS, GCP, Azure)，都能保證 100% 正常運作，不會有 "在我的電腦可以跑" 的問題。
2.  **前後端分離 (Separation of Concerns)**：
    *   **後端 (Backend)**：改用 **FastAPI** 或 **Flask** 專門處理 AI 預測，提供 API 接口 (JSON)。
    *   **前端 (Frontend)**：改用 **React**, **Vue** 或 **Next.js** 開發網頁，這樣介面可以更客製化、互動更流暢。
3.  **雲端託管 (Cloud Hosting)**：
    *   **初期**：使用 Streamlit Cloud 或 Hugging Face Spaces (免費/便宜)。
    *   **正式**：使用 **AWS (EC2/Lambda)** 或 **GCP (Cloud Run)**，配合 Load Balancer 處理大量流量。
4.  **模型優化 (Optimization)**：
    *   **格式轉換 (ONNX / TFLite)**：
        *   **比喻**：就像把 **Word** 轉成 **PDF**。
        *   **ONNX** 是一種通用格式，讓模型可以在不同硬體上加速執行；**TFLite** 則是專為手機/IoT 設備設計的輕量格式。
    *   **量化 (Quantization)**：
        *   **原理**：把模型裡的數字從「高精確度小數 (float32)」變成「簡單整數 (int8)」。
        *   **比喻**：原本記帳記到小數點後 10 位 (3.1415926535)，現在只記整數 (3)。
        *   **效果**：模型體積直接 **縮小 4 倍**，運算速度 **快 10 倍**，但準確度幾乎不變！這對手機端應用至關重要。

## 7. 總結
Day 29 我們學習了 **模型部署**。
*   AI 不應該只活在 Notebook 裡。
*   透過 **Streamlit**，我們能在幾分鐘內把模型變成產品。
*   這是讓你的價值被看見的關鍵一步！

下一章 (Day 30)，我們將迎來 **最終章：AI 總結與未來展望**。
我們將回顧這 30 天的旅程，並整理未來的學習資源 (Paper, Kaggle, MLOps)。
