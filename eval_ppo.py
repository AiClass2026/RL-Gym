"""
CarRacing-v3 PPO 评估脚本
=========================
从保存的 checkpoint 加载训练好的策略网络，在环境中运行多个 episode 并统计奖励。

支持两种动作选择模式：
    - greedy：使用策略均值（确定性策略，无探索噪声）
    - sample：从策略的高斯分布中采样（保留随机性）

评估视频自动保存到 eval_videos/{ckpt_name}/ 目录下。

用法：
    python eval_ppo.py --ckpt runs/xxx/ckpt/step10000.pt
    python eval_ppo.py --ckpt runs/xxx/ckpt/step10000.pt --mode sample --num_episodes 10
"""

import argparse
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo
from torch.distributions import Normal

from ppo.agent import ActorCritic
from ppo.utils import FrameStack, preprocess_frame

ENV_NAME = "CarRacing-v3"
FRAME_STACK_SIZE = 4
INPUT_SHAPE = (84, 84, FRAME_STACK_SIZE)


def parse_args():
    parser = argparse.ArgumentParser(description="PPO 评估 CarRacing-v3")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint 文件路径（.pt）")
    parser.add_argument(
        "--mode", type=str, default="greedy", choices=["greedy", "sample"],
        help="动作选择模式：greedy（贪心，使用均值）或 sample（从分布采样）",
    )
    parser.add_argument("--num_episodes", type=int, default=5, help="评估的 episode 数量")
    parser.add_argument("--max_steps", type=int, default=None, help="每个 episode 的最大步数（默认不限制）")
    parser.add_argument("--seed", type=int, default=0, help="环境随机种子")
    parser.add_argument("--device", type=str, default=None, help="设备，如 cuda:0 / cpu；默认自动选择")
    return parser.parse_args()


def load_policy(ckpt_path, device):
    """加载 checkpoint 并构建 ActorCritic 网络。"""
    tmp_env = gym.make(ENV_NAME, continuous=True)
    action_space = tmp_env.action_space
    num_actions = int(action_space.shape[0])
    action_min = np.array(action_space.low, dtype=np.float32)
    action_max = np.array(action_space.high, dtype=np.float32)
    tmp_env.close()

    policy = ActorCritic(INPUT_SHAPE, num_actions, action_min, action_max).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    step_idx = ckpt.get("step_idx", "unknown")
    print(f"[INFO] Loaded checkpoint: {ckpt_path}  (training step {step_idx})")
    return policy


def select_action(policy, state_tensor, mode, device):
    """
    根据模式选择动作。

    greedy: 直接使用策略均值，裁剪到合法范围。
    sample: 从高斯分布 N(mean, std) 中采样，裁剪到合法范围。
    """
    with torch.no_grad():
        mean, std, value = policy(state_tensor)
        if mode == "greedy":
            action = torch.clamp(mean, policy._action_min, policy._action_max)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, policy._action_min, policy._action_max)
    return action.cpu().numpy()[0], value.cpu().item()


def run_episode(policy, device, mode, seed, max_steps, video_dir, episode_idx):
    """运行单个 episode 并返回总奖励和步数。"""
    os.makedirs(video_dir, exist_ok=True)
    env = gym.make(
        ENV_NAME,
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda x: True,
        name_prefix=f"eval_ep{episode_idx}",
    )

    obs, _ = env.reset(seed=seed + episode_idx)
    frame_stack = FrameStack(obs, stack_size=FRAME_STACK_SIZE, preprocess_fn=preprocess_frame)

    total_reward = 0.0
    step_count = 0

    while True:
        state = frame_stack.get_state()
        state_tensor = torch.from_numpy(np.array([state])).float().to(device)

        action, _ = select_action(policy, state_tensor, mode, device)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        step_count += 1
        frame_stack.add_frame(obs)

        if terminated or truncated:
            break
        if max_steps is not None and step_count >= max_steps:
            break

    success = terminated
    env.close()
    return total_reward, step_count, success


def main():
    args = parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Action mode: {args.mode}")
    print(f"[INFO] Episodes: {args.num_episodes}")

    policy = load_policy(args.ckpt, device)

    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = os.path.join("eval_videos", f"{timestamp}_{ckpt_name}")
    print(f"[INFO] Videos: {video_dir}")

    rewards = []
    steps_list = []
    successes = []

    for ep in range(args.num_episodes):
        ep_reward, ep_steps, ep_success = run_episode(
            policy, device, args.mode, args.seed, args.max_steps,
            video_dir, ep,
        )
        rewards.append(ep_reward)
        steps_list.append(ep_steps)
        successes.append(ep_success)
        status = "SUCCESS" if ep_success else "FAIL"
        print(f"  Episode {ep + 1}/{args.num_episodes}: reward = {ep_reward:.2f}, steps = {ep_steps}, {status}")

    rewards = np.array(rewards)
    success_rate = sum(successes) / len(successes)
    print("\n" + "=" * 50)
    print(f"Results ({args.num_episodes} episodes, mode={args.mode}):")
    print(f"  Success rate: {sum(successes)}/{len(successes)} ({success_rate:.1%})")
    print(f"  Mean reward:  {rewards.mean():.2f}")
    print(f"  Std reward:   {rewards.std():.2f}")
    print(f"  Min reward:   {rewards.min():.2f}")
    print(f"  Max reward:   {rewards.max():.2f}")
    print(f"  Mean steps:   {np.mean(steps_list):.0f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
