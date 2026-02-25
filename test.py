"""
CarRacing-v3 环境快速测试脚本
==============================
本脚本用于验证 Gymnasium 赛车环境是否正确安装，并帮助理解环境的接口。

运行后会：
    1. 创建 CarRacing-v3 环境（连续动作空间）
    2. 用固定动作（全油门 + 右转）运行一个完整 episode
    3. 录制视频到 videos/ 目录
    4. 打印总奖励

用法：
    python test.py
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# ============================================================
# 创建环境
# ============================================================
# CarRacing-v3 参数说明：
#   render_mode="rgb_array"    : 以像素数组形式渲染（用于录制视频）
#   lap_complete_percent=0.95  : 跑完赛道 95% 视为完成一圈
#   domain_randomize=False     : 不随机化赛道颜色（使训练更稳定）
#   continuous=True            : 使用连续动作空间（而非离散的5个动作）
env = gym.make(
    "CarRacing-v3",
    render_mode="rgb_array",
    lap_complete_percent=0.95,
    domain_randomize=False,
    continuous=True,
)

# 用 RecordVideo 包装器自动录制每个 episode 的视频
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda x: True)

# 重置环境，获取初始观察值
obs, info = env.reset()
done = False
truncated = False
total_reward = 0.0

# ============================================================
# 定义固定动作
# ============================================================
# 连续动作空间是一个 3 维向量 [steering, gas, brake]：
#   steering : 转向，范围 [-1, 1]，-1=左转，+1=右转
#   gas      : 油门，范围 [0, 1]，0=不加速，1=全油门
#   brake    : 刹车，范围 [0, 1]，0=不刹车，1=全力刹车
# 这里使用"全右转 + 全油门 + 不刹车"作为测试动作
action = np.array([1.0, 1.0, 0.0], dtype=np.float32)

# ============================================================
# 运行 episode
# ============================================================
# 循环执行动作直到 episode 结束（terminated 或 truncated）
while not (done or truncated):
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

env.close()
print(f"回合结束，总奖励: {total_reward:.1f}")
print("视频已保存到 videos/ 文件夹")
