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
EPOCHS = 20
IMAGE_DIM = 28 * 28 # 784
H_DIM = 200         # Hidden Layer Dimension
Z_DIM = 20          # Latent Space Dimension (壓縮後的維度)

# 建立結果資料夾
os.makedirs("day34/pic", exist_ok=True)

print(f"使用裝置: {DEVICE}")

# === 2. 準備資料集 ===
dataset = torchvision.datasets.MNIST(root="day34/dataset/", train=True, transform=transforms.ToTensor(), download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === 3. 定義 VAE 模型 ===
class VAE(nn.Module):
    def __init__(self, input_dim=IMAGE_DIM, h_dim=H_DIM, z_dim=Z_DIM):
        super().__init__()
        
        # Encoder: 圖片 -> Hidden -> Mean & Variance
        self.img_2_hid = nn.Linear(input_dim, h_dim)
        self.hid_2_mu = nn.Linear(h_dim, z_dim)     # 預測平均值 (Mean)
        self.hid_2_sigma = nn.Linear(h_dim, z_dim)  # 預測變異數 (Log Variance)
        
        # Decoder: Latent(z) -> Hidden -> 圖片
        self.z_2_hid = nn.Linear(z_dim, h_dim)
        self.hid_2_img = nn.Linear(h_dim, input_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() # 輸出 0~1 (像素值)

    def encode(self, x):
        h = self.relu(self.img_2_hid(x))
        mu = self.hid_2_mu(h)
        log_sigma = self.hid_2_sigma(h)
        return mu, log_sigma

    def decode(self, z):
        h = self.relu(self.z_2_hid(z))
        return self.sigmoid(self.hid_2_img(h))

    def reparameterize(self, mu, log_sigma):
        # Reparameterization Trick: z = mu + sigma * epsilon
        std = torch.exp(0.5 * log_sigma)
        epsilon = torch.randn_like(std) # 從標準常態分佈取樣雜訊
        return mu + std * epsilon

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_sigma

# 初始化模型
model = VAE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCELoss(reduction="sum") # Reconstruction Loss

# === 4. 訓練迴圈 ===
print("開始訓練 VAE...")

for epoch in range(EPOCHS):
    total_loss = 0
    for batch_idx, (x, _) in enumerate(loader):
        x = x.view(-1, IMAGE_DIM).to(DEVICE) # 攤平
        
        # Forward
        x_reconstructed, mu, log_sigma = model(x)
        
        # Loss 計算
        # 1. Reconstruction Loss (還原誤差)
        reconst_loss = loss_fn(x_reconstructed, x)
        
        # 2. KL Divergence (分佈差異)
        # 公式: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        
        # Total Loss
        loss = reconst_loss + kl_div
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader.dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # === 5. 儲存結果 (每 5 Epoch) ===
    if (epoch + 1) % 5 == 0 or epoch == 0:
        with torch.no_grad():
            # A. 測試還原能力 (Reconstruction)
            # 拿第一批的前 8 張圖來測試
            test_x = x[:8]
            recon_x, _, _ = model(test_x)
            
            # 拼貼圖片 (上排原圖，下排還原圖)
            comparison = torch.cat([test_x.view(-1, 1, 28, 28), recon_x.view(-1, 1, 28, 28)])
            
            # 畫圖
            fig, axes = plt.subplots(2, 8, figsize=(12, 3))
            for i, ax in enumerate(axes.flat):
                ax.imshow(comparison[i][0].cpu(), cmap='gray')
                ax.axis('off')
            
            plt.suptitle(f"Reconstruction at Epoch {epoch+1} (Top: Real, Bottom: VAE)")
            plt.savefig(f"day34/pic/reconstruction_epoch_{epoch+1}.png")
            plt.close()
            
            # B. 測試生成能力 (Generation)
            # 從常態分佈隨機取樣 z，看看能生成什麼
            z_sample = torch.randn(16, Z_DIM).to(DEVICE)
            gen_x = model.decode(z_sample).view(-1, 1, 28, 28)
            
            fig, axes = plt.subplots(4, 4, figsize=(6, 6))
            for i, ax in enumerate(axes.flat):
                ax.imshow(gen_x[i][0].cpu(), cmap='gray')
                ax.axis('off')
            
            plt.suptitle(f"Generated Images at Epoch {epoch+1}")
            plt.savefig(f"day34/pic/generation_epoch_{epoch+1}.png")
            plt.close()
            
            print(f"已儲存圖片至 day34/pic/")

print("訓練完成！")
