import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os

# 嘗試匯入 gym，如果沒有則提示安裝
try:
    import gym
except ImportError:
    print("請先安裝 gym: pip install gym")
    exit()

# === 1. 超參數設定 ===
env = gym.make('CartPole-v1')
STATE_DIM = env.observation_space.shape[0]  # 4 (位置, 速度, 角度, 角速度)
ACTION_DIM = env.action_space.n             # 2 (左, 右)

LR = 0.001
GAMMA = 0.99            # 折扣因子
EPSILON_START = 1.0     # 初始探索率
EPSILON_END = 0.01      # 最終探索率
EPSILON_DECAY = 0.995   # 探索率衰減
BATCH_SIZE = 64
MEMORY_SIZE = 10000     # 經驗回放容量
TARGET_UPDATE = 10      # 每幾回合更新一次 Target Net
NUM_EPISODES = 200      # 總共玩幾回合

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 建立結果資料夾
os.makedirs("day37/pic", exist_ok=True)

# === 2. 定義 Q-Network ===
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# === 3. 定義 Agent ===
class DQNAgent:
    def __init__(self):
        self.policy_net = QNetwork(STATE_DIM, ACTION_DIM).to(DEVICE)
        self.target_net = QNetwork(STATE_DIM, ACTION_DIM).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 同步參數
        self.target_net.eval() # Target Net 不用訓練
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START

    def select_action(self, state):
        # Epsilon-Greedy
        if random.random() < self.epsilon:
            return env.action_space.sample() # 隨機探索
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state)
                return q_values.argmax().item() # 選 Q 值最大的動作

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # 隨機抽樣
        batch = random.sample(self.memory, BATCH_SIZE)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(DEVICE)
        action = torch.LongTensor(action).unsqueeze(1).to(DEVICE)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)
        next_state = torch.FloatTensor(np.array(next_state)).to(DEVICE)
        done = torch.FloatTensor(done).unsqueeze(1).to(DEVICE)

        # 計算目前的 Q(s, a)
        q_eval = self.policy_net(state).gather(1, action)

        # 計算目標 Q_target = r + gamma * max(Q_target(s', a'))
        with torch.no_grad():
            q_next = self.target_net(next_state).max(1)[0].unsqueeze(1)
            q_target = reward + (GAMMA * q_next * (1 - done))

        # Loss
        loss = nn.MSELoss()(q_eval, q_target)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

# === 4. 主訓練迴圈 ===
agent = DQNAgent()
scores = []

print(f"開始訓練 DQN on CartPole-v1 (Device: {DEVICE})...")

for i_episode in range(NUM_EPISODES):
    state, _ = env.reset() # gym 新版回傳 (state, info)
    score = 0
    
    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action) # gym 新版回傳 5 個值
        done = terminated or truncated
        
        # 修改獎勵機制：倒得越快扣分越重 (非必要，但有助於訓練)
        # 這裡用預設獎勵 (+1 per step)
        
        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()
        
        state = next_state
        score += 1
        
        if done:
            break
            
    scores.append(score)
    agent.update_epsilon()
    
    if i_episode % TARGET_UPDATE == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
    print(f"Episode {i_episode+1}/{NUM_EPISODES}, Score: {score}, Epsilon: {agent.epsilon:.2f}")

# === 5. 畫圖 ===
plt.figure(figsize=(10, 5))
plt.plot(scores)
plt.title('DQN Training Score (CartPole)')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.savefig('day37/pic/dqn_score.png')
print("訓練完成！結果圖已儲存至 day37/pic/dqn_score.png")

env.close()
