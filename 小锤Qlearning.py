import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
mpl.rcParams['font.family'] = 'KaiTi'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.unicode_minus'] = False

# 状态离散化
pos_space = np.linspace(-2.4, 2.4, 6)
vel_space = np.linspace(-3.0, 3.0, 6)
ang_space = np.linspace(-0.26, 0.26, 10)  # 更精细角度控制
ang_vel_space = np.linspace(-3.5, 3.5, 8)

def get_state_index(state):
    pos_idx = np.digitize(state[0], pos_space)
    vel_idx = np.digitize(state[1], vel_space)
    ang_idx = np.digitize(state[2], ang_space)
    ang_vel_idx = np.digitize(state[3], ang_vel_space)
    return (pos_idx, vel_idx, ang_idx, ang_vel_idx)

# 自定义奖励函数 + 提前终止机制
def custom_reward(state):
    x, x_dot, theta, theta_dot = state
    if abs(theta) > 0.26 or abs(x) > 2.4:
        return -100, True  # 失败惩罚
    # 改进后的 reward：奖励 stay alive + 惩罚离目标偏移
    penalty = (abs(theta) / 0.26)**2 + 0.2 * (abs(theta_dot)) + 0.6 * (abs(x) / 2.4)**2
    reward = 1.0 - penalty  # 保证正值范围内（0~1）
    return reward, False

# 初始化环境和 Q 表
env = gym.make('CartPole-v1')
q_table = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1,
                    len(ang_vel_space)+1, env.action_space.n), dtype=np.float32)

# 超参数
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 10000
max_steps = 500
rewards_per_episode = []
average_rewards = []  # 存储每轮的平均奖励

print("\n--- 开始训练 ---")
start_time = time.time()

for episode in range(num_episodes):
    state, _ = env.reset()
    state_idx = get_state_index(state)
    total_reward = 0
    step_count = 0

    for step in range(max_steps):
        action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[state_idx])
        new_state, _, terminated, truncated, _ = env.step(action)

        custom_r, fail_flag = custom_reward(new_state)
        done = terminated or truncated or fail_flag

        new_state_idx = get_state_index(new_state)
        max_future_q = np.max(q_table[new_state_idx])
        q_table[state_idx + (action,)] += alpha * (custom_r + gamma * max_future_q - q_table[state_idx + (action,)])

        state_idx = new_state_idx
        total_reward += custom_r
        step_count += 1

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)
    
    # 计算平均奖励 (总奖励 / 步数)
    avg_reward = total_reward / step_count if step_count > 0 else 0
    average_rewards.append(avg_reward)

    if episode % 500 == 0:
        print(f"Ep {episode}, ε={epsilon:.3f}, reward={total_reward:.1f}, avg_reward={avg_reward:.4f}")

print(f"\n训练结束，耗时: {time.time()-start_time:.2f}s")

# 平均奖励趋势图
window_size = 100
smoothed_avg = np.convolve(average_rewards, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(12,6))
plt.plot(smoothed_avg, label='当轮平均奖励')
plt.axhline(y=0.8, color='red', linestyle='--', label='成功标准')  # 根据自定义奖励函数调整标准
plt.xlabel('训练轮数')
plt.ylabel('平均奖励')
plt.title('平均奖励趋势曲线')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# 评估模型
print("\n--- 最终评估 ---")
eval_env = gym.make('CartPole-v1')
scores, success_count = [], 0
for i in range(10):
    state, _ = eval_env.reset()
    state_idx = get_state_index(state)
    score = 0
    for _ in range(max_steps):
        action = np.argmax(q_table[state_idx])
        new_state, _, terminated, truncated, _ = eval_env.step(action)
        fail_cond = abs(new_state[2]) > 0.26 or abs(new_state[0]) > 2.4
        done = terminated or truncated or fail_cond
        state_idx = get_state_index(new_state)
        score += 1
        if done:
            break
    scores.append(score)
    if score >= 195:
        success_count += 1
    print(f"评估 {i+1}: {score} 步")

print(f"\n平均维持步数: {np.mean(scores):.2f}")
print(f"成功次数（≥195步）: {success_count}/10")

# 最终演示
print("\n--- 最终策略演示 ---")
demo_env = gym.make('CartPole-v1', render_mode='human')
state, _ = demo_env.reset()
state_idx = get_state_index(state)
score = 0
for _ in range(max_steps):
    demo_env.render()
    action = np.argmax(q_table[state_idx])
    new_state, _, terminated, truncated, _ = demo_env.step(action)
    fail_cond = abs(new_state[2]) > 0.26 or abs(new_state[0]) > 2.4
    done = terminated or truncated or fail_cond
    state_idx = get_state_index(new_state)
    score += 1
    if done:
        break
demo_env.close()
print(f"最终演示得分: {score}")