import torch
from torchsummary import summary
from torchvision import models

# 1. 載入 VGG16 模型
# weights='IMAGENET1K_V1' 代表使用在 ImageNet 上預訓練好的權重
print("正在下載/載入 VGG16 模型...")
model = models.vgg16(weights='IMAGENET1K_V1')

# 2. 將模型移動到 GPU (如果有)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 3. 顯示模型摘要並存檔
print("\n=== VGG16 Model Summary ===")
# 捕捉標準輸出
import io
import sys
from contextlib import redirect_stdout

f = io.StringIO()
with redirect_stdout(f):
    summary(model, (3, 150, 150))
out = f.getvalue()

# 印出到螢幕
print(out)

# 存到檔案
with open("day45/model_summary.txt", "w", encoding="utf-8") as file:
    file.write(out)
print("摘要已儲存至 day45/model_summary.txt")
