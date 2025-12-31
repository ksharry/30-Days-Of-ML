# Day 32: 電腦視覺進階 - 物件偵測與 YOLO (Object Detection)

## 0. 前言：從「是什麼」到「在哪裡」
在 Day 19 (CNN) 我們學會了**影像分類 (Image Classification)**，機器能告訴我們「這張圖是一隻貓」。
但如果圖中有「一隻貓、兩隻狗、三個人」，而且我們想知道它們分別在圖片的**哪個位置**呢？

這就是 **物件偵測 (Object Detection)** 的任務：
1.  **Classification (分類)**：是什麼？(Cat, Dog, Person)
2.  **Localization (定位)**：在哪裡？(畫出 Bounding Box 框起來)

今天的主角是目前業界最流行、速度最快的模型：**YOLO (You Only Look Once)**。

## 1. 核心概念：YOLO (You Only Look Once)
YOLO 的名字非常霸氣：「你只需要看一次」。
早期的物件偵測 (如 R-CNN) 需要看圖片好幾千次 (提取大量候選框)，速度非常慢。
YOLO 將整張圖一次丟進神經網路，直接輸出所有物件的位置和類別，達到了 **Real-time (即時)** 的速度。

### 1.1 YOLO 演進史 (v1 - v8)
YOLO 的發展非常迅速，每一代都有顯著的進步：

| 版本 | 年份 | 主要特色 |
| :--- | :--- | :--- |
| **YOLOv1** | 2015 | **開山始祖**。將偵測視為回歸問題，速度極快 (45 FPS)，但對小物件偵測效果差。 |
| **YOLOv2** | 2016 | **Better, Faster, Stronger**。引入 Anchor Boxes (錨點框) 與 Batch Normalization，提升準確度。 |
| **YOLOv3** | 2018 | **集大成者**。引入 FPN (Feature Pyramid Networks) 多尺度偵測，大幅改善小物件偵測能力。 |
| **YOLOv4** | 2020 | **最佳化組合**。由 Alexey Bochkovskiy 接手，整合了大量 Bag of Freebies (BoF) 與 Bag of Specials (BoS) 技巧。 |
| **YOLOv5** | 2020 | **工程化落地**。Ultralytics 發布 (非論文)，改用 PyTorch 實作，極易使用，部署方便，生態系強大。 |
| **YOLOv7** | 2022 | **架構創新**。在速度與準確度上再次取得 SOTA (State-of-the-Art)，專注於模型架構的最佳化 (E-ELAN)。 |
| **YOLOv8** | 2023 | **全面升級**。Ultralytics 最新力作。改用 Anchor-Free 機制，整合了分類、偵測、分割 (Segmentation) 等多種任務。 |

### 1.2 運作流程圖 (Mermaid)
YOLO 的核心精神就是「端對端 (End-to-End)」的預測。

```mermaid
graph LR
    Input["輸入圖片"] --> CNN["CNN 特徵提取 (Backbone)"]
    CNN --> Grid["切分網格 (Grid S x S)"]
    Grid --> Predict["每個網格預測 Bounding Box + 類別"]
    Predict --> Raw["大量候選框"]
    Raw --> NMS["NMS 非極大值抑制"]
    NMS --> Output["最終輸出結果"]
```

### 1.3 網格系統 (Grid System)
YOLO 把圖片切成 $S \times S$ 的網格 (例如 $7 \times 7$)。
*   **規則**：如果一個物件的**中心點 (Center)** 落在哪個網格裡，那個網格就負責偵測這個物件。

### 1.4 輸出向量 (Output Vector)
每個網格會預測一個向量，包含：

$$
[P_c, b_x, b_y, b_w, b_h, c_1, c_2, c_3...]
$$

*   $P_c$ (Confidence)：有沒有物件？(有=1, 無=0)
*   $b_x, b_y$：中心點座標。
*   $b_w, b_h$：寬度與高度。
*   $c_1, c_2...$：是貓？是狗？是車？(類別機率)

### 1.5 YOLO 的應用場景 (Why YOLO?)
為什麼 YOLO 這麼受歡迎？因為它在 **速度 (Speed)** 與 **準確度 (Accuracy)** 之間取得了完美的平衡。
這讓它非常適合 **Real-time (即時)** 的應用：
1.  **自駕車 (Autonomous Driving)**：必須在毫秒內偵測到行人、紅綠燈、車輛，慢 0.1 秒都可能出車禍。
2.  **智慧監控 (Smart Surveillance)**：即時偵測入侵者、計算人流、偵測是否戴口罩/安全帽。
3.  **工業瑕疵檢測 (Defect Detection)**：在產線上快速掃描產品是否有裂痕或瑕疵。
4.  **運動分析 (Sports Analytics)**：即時追蹤球員與球的位置，分析戰術。

### 1.6 YOLO 為什麼這麼快？
這是面試常問的問題。
*   **傳統方法 (Two-stage, 如 Faster R-CNN)**：
    1.  先用一個演算法找出「可能由物件的區域」(Region Proposals)。
    2.  再對每個區域做分類。
    *   缺點：要做兩次工，速度慢。
*   **YOLO (One-stage)**：
    1.  直接把整張圖丟進去，同時預測「位置」和「類別」。
    2.  就像人類看照片一樣，一眼就看完，不用拿放大鏡慢慢掃描。

## 2. 關鍵評估指標
物件偵測的評估比分類複雜得多，以下三個名詞是面試必考：

### 2.1 IoU (Intersection over Union, 交集聯集比)
怎麼判斷機器畫的框 (Pred) 跟標準答案 (Truth) 準不準？
我們計算兩個框的**重疊程度**。

$$ IoU = \frac{\text{Area of Overlap (交集面積)}}{\text{Area of Union (聯集面積)}} $$

*   **IoU = 1**：完全重疊 (完美)。
*   **IoU = 0**：完全沒碰到。
*   **IoU > 0.5**：通常視為「偵測正確」的門檻。

### 2.2 NMS (Non-Maximum Suppression, 非極大值抑制)
YOLO 常常會對同一個物件預測出好幾個框 (例如一隻狗身上有 3 個框)。
**NMS 的作用就是「去蕪存菁」**：
1.  選出信心度 ($P_c$) 最高的框。
2.  把跟這個框 IoU 很高 (重疊很多) 的其他框通通刪掉。
3.  重複上述步驟，直到每個物件只剩下一個框。

### 2.3 mAP (mean Average Precision)
這是物件偵測最權威的成績單。
*   綜合考量了 **Precision (精確率)** 和 **Recall (召回率)**。
*   mAP 越高，代表模型越強。

## 3. 實戰：使用 YOLOv8 (Ultralytics)
現在最流行的版本是 **YOLOv8** (由 Ultralytics 維護)。它封裝得非常好，甚至比 Scikit-Learn 還簡單。

### 3.1 安裝
```bash
pip install ultralytics opencv-python matplotlib
```
*(注意：ultralytics 會自動安裝 PyTorch)*

### 3.2 程式碼實作
完整程式連結：[YOLO_demo.py](YOLO_demo.py)

我們使用最輕量級的 `yolov8n.pt` (Nano版) 模型，它會自動從網路下載。

```python
from ultralytics import YOLO

# 1. 載入模型
model = YOLO('yolov8n.pt')

# 2. 預測圖片 (支援 URL)
results = model('https://ultralytics.com/images/bus.jpg', save=True)

# 3. 查看結果
# 結果會自動儲存在 runs/detect/predict/ 資料夾下
```

### 3.3 執行結果範例
下圖是我們使用 `YOLO_demo.py` 執行的實際結果。
YOLO 成功在圖片中偵測到了公車、多人以及一個不明顯的停車標誌，並精準地畫出了邊框。

![YOLOv8 Detection Result](pic/32-1.png)

**詳細偵測統計**：
*   **Bus (公車)**：信心度 0.87 (非常確定)。
*   **Person (人)**：偵測到 4 位，信心度分別為 0.87, 0.85, 0.83, 0.26。
*   **Stop Sign (停車標誌)**：信心度 0.26 (雖然比較遠，但還是抓到了)。

### 3.4 進階實戰：使用攝影機即時偵測 (Webcam)
如果你想用筆電的鏡頭做即時偵測，只需要改一行程式碼：

```python
# source=0 代表使用第一個攝影機 (Webcam)
# show=True 代表直接彈出視窗顯示結果
results = model.predict(source="0", show=True)
```

### 3.5 攝影機部署建議 (Camera Deployment)
如果要將這套系統部署到實際場景 (如工廠、路口)，建議如下：

| 項目 | **測試階段 (POC)** | **正式部署 (Production)** |
| :--- | :--- | :--- |
| **攝影機硬體** | **USB Webcam** (羅技 C920 等) <br> 便宜、隨插即用，適合快速驗證。 | **IP Camera (RTSP)** <br> 透過網路傳輸，適合遠距離佈線。YOLO 支援直接讀取 RTSP 串流 (`source="rtsp://..."`)。 |
| **運算主機** | **筆電 (含 GPU)** <br> 方便攜帶與展示。 | **Edge Device (Jetson Orin)** <br> 體積小、耐高溫、低功耗，適合掛在電線桿或機台旁。 |
| **軟體優化** | **Python + PyTorch** <br> 開發速度快，但效能普通。 | **C++ + TensorRT** <br> 為了達到 30 FPS 以上的即時速度，通常會將模型轉為 TensorRT 引擎，並用 C++ 呼叫。 |

### 3.6 常見問答 (FAQ)
**Q1: 如果是影片，多久丟一張圖進去？**
這取決於你的**硬體效能**與**需求**：
*   **理想狀況**：每一幀都丟 (Frame-by-Frame)。標準影片是 30 FPS (每秒 30 張)，如果你的 GPU 夠強 (如 RTX 3090)，YOLOv8n 可以輕鬆跑到 100+ FPS，所以全丟沒問題。
*   **硬體不夠強**：跳幀處理 (Frame Skipping)。例如每 5 張圖只測 1 張 (每秒測 6 次)，中間的畫面就假設物件位置沒變，或是用簡單的演算法(Tracking)去補。

**Q2: 模型到底能認得哪些東西？(COCO Dataset)**
我們使用的 `yolov8n.pt` 是預先在 **COCO 資料集** 上訓練好的。
它認得 **80 種** 常見物件，包括：
*   **人與交通**：Person, Bicycle, Car, Motorbike, Bus, Train, Truck...
*   **動物**：Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear...
*   **生活用品**：Backpack, Umbrella, Handbag, Tie, Suitcase...
*   **電子產品**：Laptop, Mouse, Remote, Keyboard, Cell phone...

如果你想偵測這 80 種以外的東西 (例如：偵測工廠裡的瑕疵)，就需要自己收集照片來 **Fine-tune (微調)** 模型。

### 3.7 YOLOv8 模型家族選擇
除了最快的 `yolov8n.pt`，YOLOv8 還提供了一系列不同大小的模型，讓你根據需求做選擇：

| 模型代號 | 名稱 | 參數 (Params) | 速度 (Speed) | 準確度 (mAP) | 建議場景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **yolov8n.pt** | **Nano** | 3.2M | **極快** | 普通 | **手機、樹莓派**、即時性要求極高的場景。 |
| **yolov8s.pt** | **Small** | 11.2M | 快 | 佳 | **筆電 CPU**、一般 PC。CP 值最高的選擇。 |
| **yolov8m.pt** | **Medium** | 25.9M | 中 | 優 | **GPU Server**。適合需要較高準確度的商業應用。 |
| **yolov8l.pt** | **Large** | 43.7M | 慢 | 特優 | **高階 GPU**。適合遠距離偵測或小物件偵測。 |
| **yolov8x.pt** | **XLarge** | 68.2M | 極慢 | **最強** | **競賽、學術研究**。不計代價追求最高準確度。 |

*   **切換方式**：只需要更改載入名稱即可。
    ```python
    model = YOLO('yolov8m.pt') # 改用 Medium 版本
    ```

## 4. 重點複習
1.  **物件偵測 vs 影像分類**：
    *   分類：這張圖是什麼？
    *   偵測：這張圖有什麼？在哪裡？
2.  **YOLO 的核心精神**：
    *   **One-stage**：看一次就解決，速度快，適合即時系統 (如監視器、自駕車)。
3.  **IoU (交集聯集比)**：
    *   用來衡量「框得準不準」。公式：交集 / 聯集。
4.  **NMS (非極大值抑制)**：
    *   用來「刪除重複的框」，只保留最好的一個。

## 5. 下一關預告
Day 33 我們將進入 **生成式 AI (Generative AI)** 的世界。
除了讓 AI 判斷 (Discriminative)，我們還要讓 AI **創造**！
我們將從最經典的 **GAN (生成對抗網路)** 開始，看兩個神經網路如何互相博弈，創造出以假亂真的圖片。
