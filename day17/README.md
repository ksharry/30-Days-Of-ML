# Day 17: 推薦系統 (Recommender System) - 電影推薦

## 0. 歷史小故事/核心貢獻者:
推薦系統的引爆點是 **Netflix Prize (2006-2009)**。Netflix 懸賞 100 萬美元，徵求能比他們現有系統準確度提升 10% 的算法。最終，BellKor's Pragmatic Chaos 團隊獲勝，而他們的核心武器之一就是 **矩陣分解 (Matrix Factorization)**，也就是我們今天用的 SVD 技術。這場比賽徹底改變了電商和串流媒體的生態。

## 1. 資料集來源
### 資料集來源：[自製小型電影評分數據]
> 備註：為了讓數學原理更清晰，我們手動建立了一個 6 位使用者 x 4 部電影的小型矩陣。真實世界通常使用 [MovieLens](https://grouplens.org/datasets/movielens/) 資料集。

### 資料集特色與欄位介紹:
*   **User**: 使用者 (Alice, Bob, Charlie...)。
*   **Movie**: 電影 (Matrix, Titanic, Avengers, Frozen)。
*   **Rating**: 評分 (1-5 分，0 代表沒看過)。
*   **目標**: 預測使用者會給「沒看過的電影」打幾分，並推薦分數最高的。

## 2. 原理
### 核心概念：物以類聚，人以群分 (協同過濾)

#### 2.1 協同過濾 (Collaborative Filtering)
*   **User-Based**: 找出跟你看法很像的人 (鄰居)，推薦他們喜歡的東西給你。
*   **Item-Based**: 找出跟你喜歡的東西很像的物品，推薦給你。
*   **Model-Based (矩陣分解)**：我們今天用的方法。它假設使用者和電影之間存在一些「隱藏的特徵」 (Latent Features)。

#### 2.2 矩陣分解 (SVD)
我們把巨大的評分矩陣 $R$ 分解成三個小矩陣：
$$R \approx U \Sigma V^T$$
*   **$U$ (User Features)**：使用者對各個隱藏特徵的喜好程度 (例如：Alice 很愛動作片)。
*   **$V^T$ (Movie Features)**：電影含有各個隱藏特徵的程度 (例如：Matrix 是 90% 動作片)。
*   **$\Sigma$ (Weights)**：這些隱藏特徵的重要性。
*   **預測**：把這三個矩陣乘回去，就能填補原本 $R$ 矩陣中的 0 (沒看過的電影)，這些填補的值就是預測評分！

#### 2.3 國中生也能懂的案例：隱藏的分類
想像我們把電影分成兩類：**動作片** 和 **愛情片** (這就是隱藏特徵)。
*   **Alice** 給 Matrix (動作) 5 分，給 Titanic (愛情) 3 分 -> Alice 是動作片迷。
*   **Bob** 給 Avengers (動作) 5 分 -> Bob 也是動作片迷。
*   **推論**：既然 Alice 和 Bob 口味像，那 Bob 應該也會喜歡 Matrix！SVD 就是用數學自動找出這些「隱藏分類」和「口味偏好」。

## 3. 實戰
### Python 程式碼實作
完整程式連結：[Recommender_SVD.py](Recommender_SVD.py)

```python
# 關鍵程式碼：SVD 矩陣分解

# 1. 建立評分矩陣
R_df = df.pivot(index='User', columns='Movie', values='Rating').fillna(0)
R = R_df.values

# 2. 執行 SVD (k=2 代表取 2 個隱藏特徵)
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R, k=2)
sigma = np.diag(sigma)

# 3. 預測評分 (乘回去)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
```

## 4. 模型評估與視覺化
### 1. 潛在空間圖 (Latent Space)
![Latent Space](pic/17-1_Latent_Space.png)
*   **觀察**：這張圖把使用者 (藍點) 和電影 (紅叉) 畫在同一個 2D 平面上。
*   **解讀**：
    *   **左上角**：**Alice, Bob** 和 **Matrix, Avengers** 靠很近。這群是「動作片愛好者」與「動作片」。
    *   **右下角**：**Charlie** 和 **Frozen** 靠很近。這群可能是「動畫/家庭片」。
    *   **距離越近**：代表該使用者越可能喜歡該電影。
    *   **David** 在哪？他在中間偏左，離 Matrix 比較近，離 Frozen 比較遠，這跟他的評分 (Matrix=4, Frozen=1) 完全吻合！

### 2. 推薦結果 (David)
*   **已知**：David 看過 Matrix (4分), Frozen (1分)。他喜歡動作片，討厭動畫片。
*   **預測**：
    *   Titanic: **0.23 分** (雖然低，但比 Avengers 高)
    *   Avengers: **-0.09 分** (預測很低，可能因為他對 Frozen 的低分影響了整體偏好計算，或者模型認為他只愛 Matrix 這種硬派科幻，不愛超級英雄？)
    *   *註：因為數據集太小且稀疏，預測值絕對大小不重要，重要的是**相對排序**。*

## 5. 戰略總結: 非監督式學習的火箭發射之旅

### (推薦系統適用)

#### 5.1 流程一：冷啟動 (Cold Start)
*   **問題**：新使用者剛註冊，沒有任何評分。
*   **結果**：矩陣中該列全為 0，SVD 算出來也是 0，無法推薦。
*   **解法**：先推薦「熱門排行榜」，或問他喜歡什麼類型。

#### 5.2 流程二：稀疏矩陣 (Sparsity)
*   **問題**：真實世界中，一個使用者可能只看過 0.01% 的電影。
*   **結果**：矩陣幾乎都是 0，計算困難且雜訊多。
*   **解法**：使用矩陣分解 (如 SVD) 來降維，填補空缺。

#### 5.3 流程三：完美入軌 (Personalization)
*   **設定**：累積足夠的行為數據，並定期更新模型。
*   **結果**：像 Netflix 一樣，比你更懂你想看什麼！

## 6. 總結
Day 17 我們學習了 **推薦系統**。
*   這是機器學習最「變現」的應用之一。
*   **SVD** 透過找出「隱藏特徵」，能捕捉使用者和物品之間微妙的關係。
*   雖然現在有更強的深度學習推薦模型 (如 Neural CF)，但矩陣分解依然是許多系統的基石。

下一章 (Day 18)，我們將進入 **集成學習 (Ensemble Learning)** 的殿堂，學習如何結合多個模型的力量，打造出最強的分類器 —— **隨機森林 (Random Forest)**！
