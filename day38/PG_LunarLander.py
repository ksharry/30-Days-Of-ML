import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# 嘗試匯入 gymnasium
try:
    import gymnasium as gym
except ImportError:
    print("請先安裝 gymnasium: pip install gymnasium[box2d]")
    exit()

# === 1. 超參數設定 ===
# 注意: LunarLander 需要 box2d 依賴，如果安裝失敗，可以改回 CartPole-v1
ENV_NAME = 'LunarLander-v2' 
# ENV_NAME = 'CartPole-v1' 

try:
    env = gym.make(ENV_NAME)
except Exception as e:
    print(f"無法載入 {ENV_NAME}，可能是缺少 box2d。嘗試使用 CartPole-v1 代替。")
    print(f"錯誤訊息: {e}")
    ENV_NAME = 'CartPole-v1'
    env = gym.make(ENV_NAME)

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

LR = 0.002
GAMMA = 0.99
NUM_EPISODES = 500  # 訓練回合數 (LunarLander 比較難，可能需要更多回合才能收斂)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 建立結果資料夾
os.makedirs("day38/pic", exist_ok=True)

# === 2. 定義 Policy Network ===
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1) # 輸出動作的「機率」

# === 3. 定義 Agent (REINFORCE) ===
class REINFORCEAgent:
    def __init__(self):
        self.policy_net = PolicyNetwork(STATE_DIM, ACTION_DIM).to(DEVICE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.log_probs = [] # 儲存動作的 log probability
        self.rewards = []   # 儲存獎勵

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        probs = self.policy_net(state)
        
        # 根據機率分佈抽樣動作
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        # 記錄 log_prob 以便稍後計算梯度
        self.log_probs.append(m.log_prob(action))
        
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        R = 0
        policy_loss = []
        returns = []
        
        # 1. 計算回報 (Return) G_t
        # 從最後一步往前推算
        for r in self.rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(DEVICE)
        
        # 正規化回報 (穩定訓練的關鍵技巧)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # 2. 計算 Loss: -sum(log_prob * G_t)
        # 因為 PyTorch 預設是 Gradient Descent (最小化)，所以要加負號變成 Gradient Ascent (最大化獎勵)
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        # 3. Backprop
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # 清空記憶體
        self.log_probs = []
        self.rewards = []

# === 4. 主訓練迴圈 ===
agent = REINFORCEAgent()
scores = []

print(f"開始訓練 Policy Gradient on {ENV_NAME} (Device: {DEVICE})...")

for i_episode in range(NUM_EPISODES):
    state, _ = env.reset()
    score = 0
    
    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store_reward(reward)
        state = next_state
        score += 1 # 這裡簡單記錄步數作為分數 (CartPole)，如果是 LunarLander 應該用 reward 加總
        if ENV_NAME == 'LunarLander-v2':
             # 對於 LunarLander，我們用累積 reward 當作 score
             if score == 1: score = 0 # 重置
             score += reward

        if done:
            break
            
    agent.update() # 玩完一整場才更新
    scores.append(score)
    
    if (i_episode + 1) % 10 == 0:
        print(f"Episode {i_episode+1}/{NUM_EPISODES}, Score: {score:.2f}")

# === 5. 畫圖 ===
plt.figure(figsize=(10, 5))
plt.plot(scores)
plt.title(f'Policy Gradient Training Score ({ENV_NAME})')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.savefig('pic/pg_score.png')
print("訓練完成！結果圖已儲存至 pic/pg_score.png")

env.close()
