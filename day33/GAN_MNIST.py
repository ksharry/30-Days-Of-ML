import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# 設定隨機種子以確保結果可重現
torch.manual_seed(42)

# === 1. 超參數設定 (Hyperparameters) ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0002          # Learning Rate
BATCH_SIZE = 64      # 批次大小
EPOCHS = 50          # 訓練總輪數 (建議至少 50，想看好結果可設 100)
Z_DIM = 100          # 雜訊向量的維度 (Latent Dimension)
IMAGE_DIM = 28 * 28  # MNIST 圖片大小 (784)

# 建立結果資料夾
os.makedirs("day33/pic", exist_ok=True)

print(f"使用裝置: {DEVICE}")

# === 2. 準備資料集 (MNIST) ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # 將數值正規化到 [-1, 1] 之間，配合 Tanh
])

dataset = torchvision.datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === 3. 定義模型 (Generator & Discriminator) ===

# 生成器 (Generator): 雜訊 (z) -> 假圖 (Fake Image)
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01), # LeakyReLU 比 ReLU 更好，避免神經元死亡
            nn.Linear(256, img_dim),
            nn.Tanh()  # 輸出範圍 [-1, 1]，對應正規化後的圖片
        )

    def forward(self, x):
        return self.gen(x)

# 判別器 (Discriminator): 圖片 -> 真假機率 (Real/Fake)
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid() # 輸出 0~1 的機率
        )

    def forward(self, x):
        return self.disc(x)

# 初始化模型
gen = Generator(Z_DIM, IMAGE_DIM).to(DEVICE)
disc = Discriminator(IMAGE_DIM).to(DEVICE)

# 優化器 (Optimizers)
opt_gen = optim.Adam(gen.parameters(), lr=LR)
opt_disc = optim.Adam(disc.parameters(), lr=LR)

# 損失函數 (BCELoss)
criterion = nn.BCELoss()

# 固定一個雜訊向量，用來觀察訓練過程中的生成變化
fixed_noise = torch.randn(16, Z_DIM).to(DEVICE) # 產生 16 張圖

# === 4. 訓練迴圈 (Training Loop) ===
print("開始訓練 GAN...")

for epoch in range(EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        
        # --- 訓練判別器 (Discriminator) ---
        # 目標: 最大化 log(D(real)) + log(1 - D(G(z)))
        
        real = real.view(-1, IMAGE_DIM).to(DEVICE) # 攤平圖片 [Batch, 784]
        batch_size = real.shape[0]

        # 1. 放入真圖 (Real Images)
        disc_real = disc(real).view(-1) # D(x)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real)) # 真圖標籤為 1

        # 2. 放入假圖 (Fake Images)
        noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
        fake = gen(noise)
        disc_fake = disc(fake.detach()).view(-1) # D(G(z))，注意這裡要 detach，不讓梯度傳回 Generator
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # 假圖標籤為 0

        # 3. 判別器總 Loss
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # --- 訓練生成器 (Generator) ---
        # 目標: 最小化 log(1 - D(G(z))) <==> 最大化 log(D(G(z)))
        # 我們希望判別器把假圖誤判為真 (標籤為 1)
        
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output)) # 騙 D 說這是真圖 (標籤 1)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
            )

    # 每個 Epoch 結束後，儲存生成的圖片
    if (epoch + 1) % 10 == 0 or epoch == 0:
        with torch.no_grad():
            fake_images = gen(fixed_noise).reshape(-1, 1, 28, 28)
            fake_images = fake_images.cpu()
            
            # 畫圖
            fig, axes = plt.subplots(4, 4, figsize=(6, 6))
            for i, ax in enumerate(axes.flat):
                ax.imshow(fake_images[i][0], cmap='gray')
                ax.axis('off')
            
            plt.suptitle(f"Generated Images at Epoch {epoch+1}")
            save_path = f"day33/pic/epoch_{epoch+1}.png"
            plt.savefig(save_path)
            plt.close()
            print(f"已儲存生成圖片: {save_path}")

print("訓練完成！")
