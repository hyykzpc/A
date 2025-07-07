import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


# ✅ 自定义奖励函数 + 提前终止机制
def custom_reward(state):
    x, x_dot, theta, theta_dot = state
    if abs(theta) > 0.26 or abs(x) > 2.4:
        return -100, True  # 提前终止（角度或位置超过阈值）

    # 奖励计算：鼓励stay alive，同时惩罚角度和位置偏移
    penalty = (abs(theta) / 0.26)**2 + 0.2 * abs(theta_dot) + 0.6 * (abs(x) / 2.4)**2
    reward = 1.0 - penalty
    return reward, False


# ✅ 封装环境
class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.array(obs)
        reward, custom_done = custom_reward(obs)

        # 强制提前终止
        if custom_done:
            terminated = True

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return np.array(obs), info


# ✅ 记录奖励用 Callback
class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is not None:
            for i, done in enumerate(dones):
                if done:
                    info = infos[i]
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])
                        self.episode_lengths.append(info["episode"]["l"])
        return True


def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def train_ppo(total_timesteps=100000):
    env = CustomRewardEnv(gym.make("CartPole-v1"))
    model = PPO("MlpPolicy", env, verbose=1)
    reward_logger = RewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=reward_logger)
    return model, reward_logger


def test_model(model, episodes=10):
    env = CustomRewardEnv(gym.make("CartPole-v1"))
    steps_list = []
    rewards_list = []

    print(f"\n开始测试，共进行 {episodes} 次评估...\n")

    for ep in range(1, episodes + 1):
        print(f"\n>>> 第 {ep} 次测试开始")  # ← 确认是否卡在某次
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0.0

        try:
            while not done:
                action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1

                if step % 100 == 0:
                    print(f"    第 {ep} 次测试进行中，当前步数: {step}")

                if step > 2000:  # 强制终止防止卡住
                    print(f"⚠️ 第 {ep} 次测试超过 2000 步，可能卡住，强制终止")
                    break

        except Exception as e:
            print(f"❌ 第 {ep} 次测试发生异常：{e}")
            break

        steps_list.append(step)
        rewards_list.append(total_reward)
        print(f"✅ 第 {ep} 次测试完成：步数 = {step}, 总奖励 = {total_reward:.3f}")

    env.close()

    print(f"\n🔚 共完成 {len(steps_list)} 次测试")
    if len(steps_list) > 0:
        avg_steps = np.mean(steps_list)
        avg_reward = np.mean(rewards_list)
        print(f"📊 平均步数：{avg_steps:.2f}，平均奖励：{avg_reward:.3f}")




if __name__ == "__main__":
    model, reward_logger = train_ppo(total_timesteps=10000)
    test_model(model, episodes=10)
    rewards = reward_logger.episode_rewards
    if len(rewards) >= 10:
        avg_rewards = moving_average(rewards, window_size=10)
        plt.plot(avg_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Smoothed Reward (window=10)")
        plt.title("Smoothed Training Reward Curve")
        plt.grid(True)

    else:
        print("Not enough episodes to compute moving average.")

    test_model(model, episodes=10)
