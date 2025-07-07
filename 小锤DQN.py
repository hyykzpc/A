import os
import shutil
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gymnasium.wrappers import RecordVideo

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN 网络
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# 经验回放
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), actions, rewards, np.array(next_states), dones)

    def __len__(self):
        return len(self.buffer)

# 自定义奖励函数
def custom_reward(state):
    x, x_dot, theta, theta_dot = state
    if abs(theta) > 0.26 or abs(x) > 2.4:
        return -100, True  # 失败惩罚
    # 改进后的 reward：奖励 stay alive + 惩罚离目标偏移
    penalty = (abs(theta) / 0.26)**2 + 0.2 * (abs(theta_dot)) + 0.6 * (abs(x) / 2.4)**2
    reward = 1.0 - penalty  # 保证正值范围内（0~1）
    return reward, False

# 超参数和环境初始化
env = gym.make("CartPole-v1")
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

model = DQN(input_dim, output_dim).to(device)
target_model = DQN(input_dim, output_dim).to(device)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=0.01)    #学习率
replay_buffer = ReplayBuffer()

gamma = 0.99 #折扣率
batch_size = 64
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.995
target_update_freq = 10

num_episodes = 300
save_dir = "video_dqn"
os.makedirs(save_dir, exist_ok=True)

# 训练过程
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(1000):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

        next_state, _, terminated, truncated, _ = env.step(action)
        reward = custom_reward(next_state)
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

        # 训练
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states = torch.FloatTensor(np.array(states)).to(device)
            actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(np.array(next_states)).to(device)
            dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)

            q_values = model(states).gather(1, actions)
            with torch.no_grad():
                max_next_q_values = target_model(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % target_update_freq == 0:
        target_model.load_state_dict(model.state_dict())

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

env.close()

# 演示函数（支持录制视频）
def run_demo(model, record_video=False, video_folder=None, prefix="demo"):
    if record_video:
        demo_env = RecordVideo(gym.make("CartPole-v1", render_mode="rgb_array"),
                               video_folder=video_folder,
                               name_prefix=prefix)
    else:
        demo_env = gym.make("CartPole-v1")

    state, _ = demo_env.reset()
    total_steps = 0
    done = False
    while not done and total_steps < 1000:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = torch.argmax(model(state_tensor)).item()
        state, _, terminated, truncated, _ = demo_env.step(action)
        done = terminated or truncated
        total_steps += 1

    demo_env.close()
    return total_steps

