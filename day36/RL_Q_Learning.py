import numpy as np
import pandas as pd
import time

# === 1. 定義環境 (Environment) ===
# 一個簡單的 1D 世界: o---T
# o: Agent, T: Treasure, -: Ground
N_STATES = 6   # 1D 世界的長度
ACTIONS = ['left', 'right'] # 可選動作
EPSILON = 0.9   # 貪婪度 (90% 機率選最好的動作，10% 機率隨機探索)
ALPHA = 0.1     # 學習率
GAMMA = 0.9     # 遠見因子 (對未來的重視程度)
MAX_EPISODES = 15   # 玩幾回合
FRESH_TIME = 0.3    # 每一步的畫面更新時間 (秒)

def build_q_table(n_states, actions):
    # 初始化 Q-Table，全都是 0
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    return table

def choose_action(state, q_table):
    # 決定要怎麼走 (Epsilon-Greedy 策略)
    state_actions = q_table.iloc[state, :]
    
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        # 10% 機率隨機亂走 (探索)，或者一開始不知道怎麼走時
        action_name = np.random.choice(ACTIONS)
    else:
        # 90% 機率選分數最高的動作 (利用)
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(S, A):
    # 環境給回饋: 告訴 Agent 移動後發生什麼事
    # S: 當前狀態 (位置), A: 動作
    if A == 'right':    # 往右走
        if S == N_STATES - 2:   # 走到寶藏前一格 (下一步就是寶藏)
            S_ = 'terminal'     # 終止狀態
            R = 1               # 獎勵 +1
        else:
            S_ = S + 1
            R = 0
    else:   # 往左走
        R = 0
        if S == 0:
            S_ = S  # 撞牆 (原地不動)
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter):
    # 更新畫面 (CLI 介面)
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T'
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    # 主迴圈
    q_table = build_q_table(N_STATES, ACTIONS)
    
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0   # 回合開始，回到起點
        is_terminated = False
        update_env(S, episode, step_counter)
        
        while not is_terminated:
            # 1. 選動作
            A = choose_action(S, q_table)
            
            # 2. 執行動作，得到下一個狀態和獎勵
            S_, R = get_env_feedback(S, A)
            
            # 3. 估算 (預測)
            q_predict = q_table.loc[S, A]
            
            # 4. 現實 (目標)
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # 下一狀態的最大 Q 值
            else:
                q_target = R     # 下一狀態是終點，沒有未來了
                is_terminated = True    # 結束這回合
            
            # 5. 更新 Q-Table
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            
            # 6. 移動到下一個狀態
            S = S_
            
            update_env(S, episode, step_counter+1)
            step_counter += 1
            
    return q_table

if __name__ == "__main__":
    print("開始尋寶遊戲 (Q-Learning)...")
    print("環境: o---T (o: Agent, T: Treasure)")
    q_table = rl()
    print("\n\n最終的 Q-Table (作弊小抄):")
    print(q_table)
