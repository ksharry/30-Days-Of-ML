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
    Input[輸入圖片] --> CNN[CNN 特徵提取 (Backbone)]
    CNN --> Grid[切分網格 (Grid S x S)]
    Grid --> Predict[每個網格預測 Bounding Box + 類別]
    Predict --> Raw[大量候選框]
    Raw --> NMS[NMS 非極大值抑制]
    NMS --> Output[最終輸出結果]
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
