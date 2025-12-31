from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

def main():
    print("=== Day 32: YOLOv8 Object Detection Demo ===")
    
    # 1. 載入預訓練模型 (Load Pre-trained Model)
    # 第一次執行會自動下載 yolov8n.pt (Nano 版本，速度最快)
    print("正在載入 YOLOv8 Nano 模型...")
    model = YOLO('yolov8n.pt') 

    # 2. 設定測試圖片來源
    # 這裡我們使用一張網路上常見的測試圖 (巴士與人)
    image_url = 'https://ultralytics.com/images/bus.jpg'
    
    print(f"正在對圖片進行物件偵測: {image_url}")
    
    # 3. 進行預測 (Inference)
    # save=True 會把結果存成圖片
    results = model(image_url, save=True, project='runs/detect', name='day32_demo', exist_ok=True)

    # 4. 顯示結果
    # YOLO 會把結果存在 runs/detect/day32_demo/bus.jpg
    result_path = os.path.join('runs', 'detect', 'day32_demo', 'bus.jpg')
    
    if os.path.exists(result_path):
        print(f"\n偵測完成！結果已儲存至: {result_path}")
        
        # 使用 matplotlib 顯示結果 (因為在 server 環境可能無法直接彈出視窗)
        img = cv2.imread(result_path)
        # OpenCV 讀進來是 BGR，要轉成 RGB 顯示才正常
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title("YOLOv8 Detection Result")
        plt.show()
        
        print("圖片已顯示。")
    else:
        print("找不到結果圖片，請檢查執行過程是否有錯誤。")

    # 5. 列出偵測到的物件
    print("\n=== 偵測統計 ===")
    # results 是一個 list，因為我們只傳了一張圖，所以拿 results[0]
    result = results[0]
    
    # result.boxes 包含了所有的偵測框
    for box in result.boxes:
        class_id = int(box.cls[0]) # 類別 ID
        class_name = model.names[class_id] # 類別名稱
        conf = float(box.conf[0]) # 信心度
        
        print(f"找到: {class_name} (信心度: {conf:.2f})")

if __name__ == "__main__":
    main()
