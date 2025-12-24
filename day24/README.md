# Day 24: 神經網路訓練 - MNIST 手寫數字辨識

## 0. 歷史小故事/核心貢獻者:
**MNIST (Modified National Institute of Standards and Technology database)** 是深度學習界的 "Hello World"。
它由 **Yann LeCun** 等人於 1998 年整理發布。在那個年代，能讓電腦自動辨識手寫支票上的數字，是一項了不起的技術。
如果你提出一個新的圖像識別演算法，大家會問的第一個問題通常是：「它在 MNIST 上跑幾分？」

## 1. 資料集來源
### 資料集來源：[MNIST Database](http://yann.lecun.com/exdb/mnist/)
> 備註：Keras 已內建此資料集，程式會自動下載。

### 資料集特色與欄位介紹:
這是一個包含 0~9 手寫數字的灰階圖片資料集。
*   **數量**：訓練集 60,000 筆，測試集 10,000 筆。
*   **格式**：28x28 像素 (Pixel)，灰階值 0 (白) ~ 255 (黑)。
*   **目標 (Target)**：0, 1, 2, ..., 9 (共 10 類)。

## 2. 原理
### 核心概念：神經網路是如何學習的？ (Training Loop)
訓練一個神經網路，其實就是不斷重複以下四個步驟，直到它學會為止：

1.  **前向傳播 (Forward Propagation)**：
    *   資料從輸入層進去，經過層層神經元 (加權總和+激活)，最後算出預測結果。
    *   *比喻：學生寫考卷，寫出答案。*

2.  **計算損失 (Calculate Loss)**：
    *   比較「預測結果」和「真實答案」的差距。常用的損失函數是 **Cross Entropy (交叉熵)**。
    *   *比喻：老師改考卷，算出考了幾分 (或錯了幾題)。*

3.  **反向傳播 (Backpropagation)**：
    *   這是深度學習的靈魂！根據誤差，從最後一層往回推，計算每個神經元對誤差「貢獻」了多少 (梯度 Gradient)。
    *   *比喻：檢討考卷，發現是哪一個觀念錯了，導致最後答案寫錯。*

4.  **更新權重 (Update Weights - Optimizer)**：
    *   根據梯度，調整神經元的權重 (Weight) 和偏差 (Bias)。我們使用 **Adam** 優化器。
    *   *比喻：學生修正腦中的觀念，下次就不會再錯了。*

### 模型架構 (Architecture)
我們使用一個簡單的 **MLP (多層感知機)**：
*   **Input**: 28x28 圖片 -> **Flatten** -> 784 維向量。
*   **Hidden**: 128 個神經元 (**ReLU**)。
*   **Output**: 10 個神經元 (**Softmax**，輸出 0~9 的機率)。

## 3. 實戰
### Python 程式碼實作
完整程式連結：[DL_MNIST_Training.py](DL_MNIST_Training.py)

```python
# 關鍵程式碼：建立與訓練模型

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# 1. 建立模型
model = Sequential([
    Flatten(input_shape=(28, 28)),      # 拉平
    Dense(128, activation='relu'),      # 隱藏層
    Dense(10, activation='softmax')     # 輸出層 (10類)
])

# 2. 編譯模型 (設定學習方式)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 訓練模型 (開始寫考卷、改考卷、修正觀念)
model.fit(X_train, y_train, epochs=10)
```

## 4. 模型評估與視覺化
### 1. 訓練過程 (History)
![Training History](pic/24-1_Training_History.png)
*   **Loss (左圖)**：隨著訓練次數 (Epoch) 增加，誤差越來越小。
*   **Accuracy (右圖)**：準確率越來越高，最終在測試集上達到了約 **97.8%** 的準確率。

### 2. 預測結果展示 (Predictions)
![Predictions](pic/24-2_Predictions.png)
*   **觀察**：
    *   模型成功辨識出了大部分的數字。
    *   標題顯示了 `Pred` (預測值) 和 `True` (真實值)。
    *   如果有寫得很醜的字 (例如像 1 的 7)，模型可能會認錯，這時候就需要更強的模型 (如 CNN)。

## 5. 戰略總結: 深度學習的標準 SOP

### (Deep Learning 適用)

#### 5.1 流程一：資料預處理 (Preprocessing)
*   **設定**：將圖片像素除以 255，縮放到 0~1 之間。
*   **目的**：**正規化 (Normalization)** 能讓梯度下降跑得更順暢，加速收斂。

#### 5.2 流程二：定義模型 (Model Definition)
*   **設定**：Input -> Flatten -> Dense (ReLU) -> Dense (Softmax)。
*   **原則**：對於影像問題，MLP 破壞了空間結構 (把 2D 變 1D)，所以效果有極限。明天我們要學的 CNN 會解決這個問題。

#### 5.3 流程三：編譯與訓練 (Compile & Fit)
*   **設定**：Optimizer=Adam, Loss=CrossEntropy。
*   **目的**：讓模型自動透過反向傳播算法，找到最佳的權重參數。

## 6. 總結
Day 24 我們完成了深度學習的 "Hello World"。
*   我們訓練了一個 MLP 來辨識手寫數字，準確率達到 97.8%。
*   我們理解了 **Forward -> Loss -> Backward -> Update** 的訓練循環。
*   雖然 MLP 很厲害，但它把圖片「壓扁」了，丟失了像素之間的空間關係。

下一章 (Day 25)，我們將引入 **CNN (卷積神經網路)**，它能像人類眼睛一樣，看懂圖片的「特徵」！
