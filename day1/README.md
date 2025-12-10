# Day 01 起點與地圖：人工智慧地圖與演算法選擇指南

## 前言：為什麼要看地圖？

大家好，我是 Harry Chang。為了回憶人工智慧學校所學，歡迎來到我的 30 天 AI 實戰筆記。

在接下來的 30 天，我們將進行一場「演算法的練習」。從最經典的線性回歸，一路實作到最新的生成式 AI。但在我們急著打開 Python 寫 Code 之前，必須先搞清楚我們身在何處。

很多人學 AI 會迷路，是因為分不清楚 Buzzwords（流行術語）之間的關係，或者拿到資料後不知道該選哪把「武器」。今天這篇文章，就是這趟旅程的導航地圖。

---

## 1. 俄羅斯娃娃：AI、ML 與 DL 的關係

首先，我們要破除迷思。人工智慧 (AI)、機器學習 (ML) 與深度學習 (DL) 並不是三個獨立的技術，它們是層層包裹的關係。

![https://ithelp.ithome.com.tw/upload/images/20251210/20161788WfwWpH7iaB.png](https://ithelp.ithome.com.tw/upload/images/20251210/20161788WfwWpH7iaB.png)

* **人工智慧 (Artificial Intelligence, AI)**：最外層的大圈圈。泛指所有展現智慧的機器技術。早期的 AI 甚至不需要「學習」，只要人類寫好 `if-else` 規則（Rule-based）就算。
* **機器學習 (Machine Learning, ML)**：AI 的子集。核心在於**「讓機器從資料中找規律」**，而不是人類手寫規則。我們前兩週的重點（如回歸、決策樹、SVM）都在這裡。
* **深度學習 (Deep Learning, DL)**：ML 的子集。特指使用**「多層神經網路」**的技術。當資料量大到傳統 ML 跑不動時，就是 DL 發揮威力的時候（如 CNN、RNN）。

---

## 2. 迷路時的指南針：演算法選擇地圖 (Cheat Sheets)

新手最常問的問題是：「我有資料了，但我到底該用 SVM 還是 Random Forest？」

不用擲筊，前人已經幫我們畫好了地圖。以下這兩張圖被譽為 AI 界的「藏寶圖」，請務必按右鍵收藏。

### 第一張：Scikit-Learn 演算法地圖

這是 Python 最常用的機器學習套件 `scikit-learn` 官方繪製的流程圖。它的邏輯非常工程師思維：從 `Start` 開始，根據你的資料量大小、資料型態，一步步引導你走到適合的演算法。

![Scikit-Learn Algorithm Cheat Sheet](https://ithelp.ithome.com.tw/upload/images/20251210/20161788yTvqdqPRaB.jpg)
![Scikit-Learn Algorithm Cheat Sheet](https://github.com/ksharry/30-Days-Of-ML/blob/main/day1/pic/1-2.jpg?raw=true)
*(圖片來源：[Scikit-Learn Official Documentation](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html))*

**怎麼看這張圖？**
1.  **Start**：資料量是否超過 50 筆？（太少就別跑 ML 了，先去收資料！）
2.  **Category vs Quantity**：你要預測的是「類別」（是貓還是狗？）還是「數值」（房價是多少？）
3.  如果不準確 (Not Working)，圖表會建議你換下一條路（例如從 Linear SVC 換到 KNN）。



### 第二張：微軟 Azure 機器學習演算法圖譜

如果上一張是流程圖，這張微軟製作的圖表則像是一張「功能菜單」。

![Microsoft Azure Machine Learning Algorithm Cheat Sheet](https://learn.microsoft.com/en-us/azure/machine-learning/media/algorithm-cheat-sheet/machine-learning-algorithm-cheat-sheet.png)
*(圖片來源：[Microsoft Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/algorithm-cheat-sheet?view=azureml-api-2))*

它將演算法依照**功能**分為四大象限，我們這 30 天將會逐一攻克：

* **回歸 (Regression)**：預測數值（如：明天股價）。
* **分類 (Classification)**：預測類別（如：是否為垃圾郵件）。
* **分群 (Clustering)**：沒有標準答案，將相似的資料聚在一起（如：客戶分群）。
* **異常檢測 (Anomaly Detection)**：抓出不一樣的資料（如：盜刷偵測）。

---

## 3. 30 天的旅程預告

有了地圖，接下來我們就需要「武器」。這 30 天，我將不會花時間教你怎麼安裝 Python（假設大家都會了），我們將採用**「一天一演算法 + 實作」**的模式。

明天 Day 02，我們將直接挑戰機器學習的始祖：**線性回歸 (Linear Regression)**。

我們明天見！

