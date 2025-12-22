# Day 09: 決策樹 (Decision Tree) - 鐵達尼號生存預測

## 0. 歷史小故事/核心貢獻者:
**Ross Quinlan** 在 1986 年提出了 ID3 演算法，後來演進為 C4.5，奠定了決策樹的基礎。決策樹的靈感來自於人類的思考方式：我們在做決定時，往往會問一系列的「是/否」問題 (例如：外面下雨嗎？是 -> 帶傘；否 -> 不帶)。這種直觀的邏輯讓決策樹成為最容易被人類理解 (Explainable AI) 的模型之一。

## 1. 資料集來源
### 資料集來源：[Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
> 備註：這是 Kaggle 上最著名的入門競賽。任務是預測鐵達尼號乘客是否倖存。

### 資料集特色與欄位介紹:
這是一個二元分類問題 (Binary Classification)，但包含了數值與類別特徵，且有缺失值，非常適合練習資料前處理。

**欄位說明**：
*   **Survived (目標 y)**: 是否倖存 (0=No, 1=Yes)。
*   **Pclass**: 艙等 (1=頭等艙, 2=二等艙, 3=三等艙)。這通常代表社會地位。
*   **Sex**: 性別 (male, female)。
*   **Age**: 年齡。
*   **SibSp**: 兄弟姊妹/配偶數。
*   **Parch**: 父母/小孩數。
*   **Fare**: 票價。
*   **Embarked**: 登船港口 (C=Cherbourg, Q=Queenstown, S=Southampton)。

### 資料清理
1.  **缺失值填補**：
    *   `Age`：使用中位數填補 (避免極端值影響)。
    *   `Embarked`：使用眾數 (出現最多次的港口) 填補。
2.  **類別轉數字 (Label Encoding)**：
    *   決策樹雖然理論上可處理文字類別，但 sklearn 的實作需要數值輸入。
    *   `Sex`：male -> 1, female -> 0。
    *   `Embarked`：C -> 0, Q -> 1, S -> 2。

## 2. 原理
### 核心概念：如何畫出這棵樹？
決策樹的生長過程就是不斷地問問題，把資料切分得越來越「純 (Pure)」。

1.  **資訊增益 (Information Gain) 與 熵 (Entropy)**：
    *   **熵 (Entropy)**：衡量混亂程度。如果一堆人裡面有一半活一半死，熵最大 (最混亂)；如果全部都活，熵最小 (最純)。
    *   **目標**：每次切分都要讓子節點的熵變小 (變純)。
    *   **公式**：
        $$Entropy(S) = -p_+ \log_2 p_+ - p_- \log_2 p_-$$

2.  **基尼係數 (Gini Impurity)**：
    *   另一種衡量不純度的方法 (sklearn 預設使用)。計算比較快，效果跟 Entropy 差不多。
    *   **公式**：
        $$Gini = 1 - \sum (p_i)^2$$

### 決策樹如何挑選特徵？ (How Split Works)
你可能會好奇：「為什麼根節點選的是 **Sex (性別)** 而不是 Age 或 Pclass？」
這是一個 **窮舉搜尋 (Greedy Search)** 的過程：

1.  **第一步**：模型會把所有特徵 (Sex, Age, Pclass...) 都拿來試切一遍。
2.  **第二步**：計算切分後的「資訊增益 (Information Gain)」。
    *   $Gain = \text{切分前的熵} - \text{切分後的加權平均熵}$
    *   簡單說就是：**切完之後，混亂程度下降了多少？**
3.  **第三步**：比較結果。
    *   如果依 Sex 切分，Gain = 0.2
    *   如果依 Pclass 切分，Gain = 0.1
    *   如果依 Age 切分，Gain = 0.05
4.  **結論**：Sex 的 Gain 最大 (下降最多)，所以選 Sex 當作根節點！
5.  **重複**：接著在左右子節點，重複上述步驟，直到達到停止條件 (如 max_depth)。

也就是說，電腦真的**每個特徵、每個切分點**都算過了一遍，才決定出這棵樹的長相。

3.  **剪枝 (Pruning)**：
    *   如果樹長得太深 (問太多細節)，會導致 Overfitting (把雜訊都學起來)。
    *   我們設定 `max_depth=3` 來限制樹的高度，這就是一種「預剪枝」。

## 3. 實戰
### Python 程式碼實作
完整程式連結：[Decision_Tree_Titanic.py](Decision_Tree_Titanic.py)

```python
# 關鍵程式碼：訓練決策樹
from sklearn.tree import DecisionTreeClassifier
# criterion='entropy': 使用資訊增益
# max_depth=3: 限制樹深，避免過擬合
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
classifier.fit(X_train, y_train)
```

## 4. 模型評估
### 若為分類模型 (Classification)
*   **混淆矩陣圖**：
    ![Confusion Matrix](pic/9-1_Confusion_Matrix.png)
*   **指標數字**：
    *   **Accuracy (準確率)**: `0.8206` (預測正確率約 82%)。
    *   **Precision (Survived=1)**: `0.79` (預測會活的人，79% 真的活了)。
    *   **Recall (Survived=1)**: `0.71` (真的活下來的人，抓到了 71%)。

*   **決策樹視覺化 (Visualization)**：
    ![Decision Tree](pic/9-2_Decision_Tree.png)
    *   **解讀這棵樹**：
        *   **Root Node (根節點)**：`Sex <= 0.5` (即 Female)。
            *   **True (左邊)**：女性。生存機率大幅提升。
            *   **False (右邊)**：男性。大部分都沒活下來。
        *   **第二層**：
            *   如果是女性，接著看 `Pclass` (艙等)。頭等艙女性生存率最高。
            *   如果是男性，接著看 `Age`。小孩 (Age <= 6.5) 有較高生存機會。
    *   **結論**：**"Lady and Children First"** (婦孺優先) 這句歷史名言，被決策樹完美地學到了！

*   **特徵重要性 (Feature Importance)**：
    ![Feature Importance](pic/9-3_Feature_Importance.png)
    *   **Sex (性別)**：最重要的特徵 (壓倒性勝利)。
    *   **Pclass (艙等)**：第二重要。
    *   **Age (年齡)**：第三重要。

## 5. 戰略總結:模型訓練的火箭發射之旅

### (回歸與監督式學習適用day2-12)
引用大師-吳恩達教授的 Rocket 進行說明 Bias vs Variance：
![rocket](https://github.com/ksharry/30-Days-Of-ML/blob/main/day02/pic/2-4_Rocket.jpg?raw=true)

#### 5.1 流程一：推力不足，無法升空 (Underfitting 迴圈)
*   **設定**：`max_depth=1` (只有一層，稱為 Decision Stump)。
*   **結果**：只能問一個問題 (例如性別)，雖然能分出一部分，但無法捕捉複雜的規則 (如「三等艙的女性」可能比較危險)。

#### 5.2 流程二：動力太強，失控亂飛 (Overfitting 迴圈)
*   **設定**：不限制深度 (`max_depth=None`)。
*   **結果**：樹會長得非常茂盛，甚至去記住某個 22 歲買了 7.25 元票的人活了下來。訓練集準確率可能 100%，但測試集會很慘。

#### 5.3 流程三：完美入軌 (The Sweet Spot)
*   **設定**：`max_depth=3` 或 `max_depth=5`。
*   **結果**：抓住了主要規律 (性別、艙等、年齡)，忽略了瑣碎的雜訊，達到最佳的泛化能力。

## 6. 總結
Day 09 我們學習了 **決策樹 (Decision Tree)**。
*   **可解釋性 (Explainability)**：這是決策樹最強的地方，你可以畫出圖來告訴老闆為什麼模型會這樣判斷。
*   **特徵選擇**：決策樹會自動挑選重要的特徵 (Root Node 通常就是最重要的)。
*   **過擬合**：決策樹非常容易過擬合，一定要記得限制深度 (Pruning)。
下一章 (Day 10)，我們將進入 **支持向量機 (SVM)**，這是一個數學味很重，但在深度學習流行之前稱霸武林的演算法！
