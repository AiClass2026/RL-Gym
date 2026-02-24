import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# 使用连续动作空间，以便同时踩油门和向右转向
env = gym.make(
    "CarRacing-v3",
    render_mode="rgb_array",
    lap_complete_percent=0.95,
    domain_randomize=False,
    continuous=True,
)

# 录制视频到 videos 文件夹
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda x: True)

obs, info = env.reset()
done = False
truncated = False
total_reward = 0.0

# 连续动作: [steering, gas, brake]
# steering: -1 左, +1 右
# gas: 0~1 油门
# brake: 0~1 刹车
# 一直踩油门 + 向右转向
action = np.array([1.0, 1.0, 0.0], dtype=np.float32)  # 全右转 + 全油门 + 不刹车

while not (done or truncated):
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

env.close()
print(f"回合结束，总奖励: {total_reward:.1f}")
print("视频已保存到 videos/ 文件夹")
