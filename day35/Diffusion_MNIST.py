import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# 設定隨機種子
torch.manual_seed(42)

# === 1. 超參數設定 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 3  # 為了示範，只跑 3 個 Epoch (完整訓練需要非常久)
TIMESTEPS = 500 # 加噪/去噪的總步數 (T)
IMAGE_DIM = 28 * 28

# 建立結果資料夾
os.makedirs("day35/pic", exist_ok=True)

print(f"使用裝置: {DEVICE}")

# === 2. 準備資料集 ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x - 0.5) * 2) # 正規化到 [-1, 1]
])

dataset = torchvision.datasets.MNIST(root="day35/dataset/", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === 3. 定義 Diffusion 核心組件 ===

# A. Noise Scheduler (負責計算 alpha, beta)
class NoiseScheduler:
    def __init__(self, timesteps=TIMESTEPS):
        self.timesteps = timesteps
        # 定義 beta (雜訊量)，從 0.0001 到 0.02 線性增加
        self.betas = torch.linspace(0.0001, 0.02, timesteps).to(DEVICE)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) # 連乘積
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    # Forward Process: 加噪 (x0 -> xt)
    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0).to(DEVICE)
        
        # 公式: xt = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise, noise

# B. U-Net (負責預測雜訊)
# 這裡用一個簡化版的 U-Net (只有 3 層)
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Downsample (Encoder)
        self.down1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.down2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        
        # Time Embedding (把時間 t 告訴模型)
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Upsample (Decoder)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(32, 1, 2, stride=2)) # 輸出 1 channel (雜訊)

    def forward(self, x, t):
        # 處理時間 t
        t = t.float().view(-1, 1).to(DEVICE) / TIMESTEPS
        t_emb = self.time_embed(t).view(-1, 64, 1, 1)
        
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        
        # 融合時間資訊
        x2 = x2 + t_emb
        
        # Decoder
        x = self.up1(x2)
        x = x + x1 # Skip Connection (U-Net 的特徵)
        x = self.up2(x)
        return x

# 初始化
scheduler = NoiseScheduler()
model = SimpleUNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss() # 預測雜訊 vs 真實雜訊

# === 4. 訓練迴圈 ===
print("開始訓練 Diffusion Model (DDPM)...")

for epoch in range(EPOCHS):
    total_loss = 0
    for batch_idx, (x0, _) in enumerate(loader):
        x0 = x0.to(DEVICE)
        batch_size = x0.shape[0]
        
        # 1. 隨機選時間 t
        t = torch.randint(0, TIMESTEPS, (batch_size,)).to(DEVICE)
        
        # 2. 加噪 (Forward)
        noise = torch.randn_like(x0).to(DEVICE)
        xt, _ = scheduler.add_noise(x0, t, noise)
        
        # 3. 預測雜訊 (Predict Noise)
        noise_pred = model(xt, t)
        
        # 4. 算 Loss
        loss = criterion(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(loader):.4f}")

# === 5. 生成過程 (Sampling) ===
print("開始生成圖片 (Reverse Process)...")
model.eval()
with torch.no_grad():
    # 1. 從純雜訊開始 (xT)
    x = torch.randn(16, 1, 28, 28).to(DEVICE)
    
    # 2. 一步步去噪 (T -> 0)
    for i in reversed(range(TIMESTEPS)):
        t = torch.full((16,), i, dtype=torch.long).to(DEVICE)
        
        # 預測雜訊
        noise_pred = model(x, t)
        
        # 去噪公式 (簡化版): x_{t-1} = (x_t - beta * noise_pred) / sqrt(alpha) + sigma * z
        alpha = scheduler.alphas[i]
        beta = scheduler.betas[i]
        alpha_cumprod = scheduler.alphas_cumprod[i]
        sqrt_one_minus_alpha_cumprod = scheduler.sqrt_one_minus_alphas_cumprod[i]
        
        # 這裡只做最簡單的去噪演示 (不加 sigma * z)
        x = (x - (beta / sqrt_one_minus_alpha_cumprod) * noise_pred) / torch.sqrt(alpha)
        
    # 畫圖
    x = x.clamp(-1, 1) # 限制數值範圍
    x = (x + 1) / 2    # 還原到 [0, 1]
    
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i][0].cpu(), cmap='gray')
        ax.axis('off')
    
    plt.suptitle("Generated Images by Diffusion")
    plt.savefig("day35/pic/diffusion_result.png")
    print("已儲存生成圖片: day35/pic/diffusion_result.png")

print("完成！")
