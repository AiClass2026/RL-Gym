"""
CarRacing-v3 PPO 训练脚本
使用 AsyncVectorEnv 并行环境 + TensorBoard
"""
import os
import argparse
from datetime import datetime

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm

from ppo import PPOAgent, FrameStack, preprocess_frame, compute_gae

ENV_NAME = "CarRacing-v3"
FRAME_STACK_SIZE = 4
INPUT_SHAPE = (84, 84, FRAME_STACK_SIZE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO 训练 CarRacing-v3"
    )
    parser.add_argument("--initial_lr", type=float, default=3e-4, help="初始学习率")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="折扣因子 γ")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE 的 λ 参数")
    parser.add_argument("--ppo_epsilon", type=float, default=0.2, help="PPO 裁剪参数 ε")
    parser.add_argument("--value_scale", type=float, default=0.5, help="价值损失缩放系数")
    parser.add_argument("--entropy_scale", type=float, default=0.01, help="熵损失缩放系数")
    parser.add_argument("--horizon", type=int, default=64, help="每次 PPO 更新前，每个 env 采样步数")
    parser.add_argument("--num_epochs", type=int, default=10, help="每次 rollout 后的 PPO 更新轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="mini-batch 大小")
    parser.add_argument("--num_envs", type=int, default=64, help="并行环境数量")
    parser.add_argument("--model_name", type=str, default="CarRacing-v3", help="模型名称，用于 run 目录后缀")
    parser.add_argument("--save_interval", type=int, default=10000, help="保存 checkpoint 的间隔（训练步数）")
    parser.add_argument("--eval_interval", type=int, default=2000, help="评估的间隔（训练步数）")
    parser.add_argument("--sync", action="store_true", help="使用 SyncVectorEnv 替代 Async（调试用）")
    parser.add_argument("--eval_max_steps", type=int, default=None, help="每次评估的最大环境步数（默认不限制）")
    parser.add_argument("--max_train_steps", type=int, default=280000, help="最大训练步数，达到后自动停止")
    parser.add_argument("--device", type=str, default=None, help="设备，如 cuda:0 / cuda / cpu；默认自动选 cuda")
    parser.add_argument("--run_id", type=str, default=None, help="运行唯一标识，多进程时避免目录冲突；默认用 PID")
    return parser.parse_args()


def make_env():
    return gym.make(
        ENV_NAME,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )


def evaluate(agent, frame_stack_size, step_idx=None, max_steps=None):
    """评估 agent，返回总奖励和 value error。每次评估都会录制视频，以训练步数命名。"""
    test_env = gym.make(
        ENV_NAME,
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )
    # 每次评估都录制视频，以训练步数命名（如 step_1000-episode-0.mp4）
    name_prefix = f"step_{step_idx}" if step_idx is not None else "eval"
    test_env = RecordVideo(
        test_env,
        video_folder=agent.video_dir,
        episode_trigger=lambda x: True,
        name_prefix=name_prefix,
    )

    obs, _ = test_env.reset(seed=0)
    frame_stack = FrameStack(obs, stack_size=frame_stack_size, preprocess_fn=preprocess_frame)

    total_reward = 0.0
    values_list = []
    rewards_list = []
    dones_list = []

    step_count = 0
    while True:
        state = frame_stack.get_state()
        action, value = agent.predict([state], use_old_policy=True, greedy=False)
        obs, reward, terminated, truncated, _ = test_env.step(action[0])
        total_reward += float(reward)
        done = terminated or truncated
        step_count += 1

        values_list.append(value[0])
        rewards_list.append(reward)
        dones_list.append(done)

        frame_stack.add_frame(obs)

        if done or (max_steps is not None and step_count >= max_steps):
            break

    # 计算 value error
    last_value = agent.predict([frame_stack.get_state()], use_old_policy=True, greedy=False)[1][0]
    returns = []
    R = last_value
    for i in reversed(range(len(rewards_list))):
        R = rewards_list[i] + (1.0 - float(dones_list[i])) * 0.99 * R
        returns.append(R)
    returns = np.array(list(reversed(returns)))
    value_error = np.mean(np.square(np.array(values_list) - returns))

    test_env.close()
    return total_reward, value_error


def train():
    args = parse_args()

    def lr_scheduler(step_idx):
        return args.initial_lr * 0.85 ** (step_idx // 10000)

    # 创建测试环境以获取 action space
    test_env = gym.make(
        ENV_NAME,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )
    action_space = test_env.action_space
    num_actions = int(action_space.shape[0])
    action_min = np.array(action_space.low, dtype=np.float32)
    action_max = np.array(action_space.high, dtype=np.float32)
    test_env.close()

    # 创建以时间戳为前缀的 run 目录，多进程时用 run_id 区分
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id if args.run_id is not None else str(os.getpid())
    run_dir = os.path.join("./runs", f"{timestamp}_{run_id}_{args.model_name}")
    print(f"[INFO] Run directory: {run_dir}")

    # 创建 agent
    agent = PPOAgent(
        INPUT_SHAPE,
        num_actions,
        action_min,
        action_max,
        epsilon=args.ppo_epsilon,
        value_scale=args.value_scale,
        entropy_scale=args.entropy_scale,
        run_dir=run_dir,
        device=args.device,
    )

    # 创建并行环境 (AsyncVectorEnv 或 SyncVectorEnv)
    vec_mode = "sync" if args.sync else "async"
    envs = gym.make_vec(
        ENV_NAME,
        num_envs=args.num_envs,
        vectorization_mode=vec_mode,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )

    # 初始化
    obs, _ = envs.reset()
    frame_stacks = [
        FrameStack(obs[i], stack_size=FRAME_STACK_SIZE, preprocess_fn=preprocess_frame)
        for i in range(args.num_envs)
    ]

    max_train_steps = args.max_train_steps
    print("[INFO] Training loop started" + (f" (max {max_train_steps} steps)" if max_train_steps else ""))

    pbar = tqdm(
        total=max_train_steps,
        unit="step",
        unit_scale=False,
        desc="Training",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    try:
        while True:
            if max_train_steps is not None and agent.step_idx >= max_train_steps:
                break

            states, taken_actions, values, rewards, dones = [], [], [], [], []

            for _ in range(args.horizon):
                states_t = [frame_stacks[i].get_state() for i in range(args.num_envs)]
                actions_t, values_t = agent.predict(states_t, use_old_policy=True, greedy=False)

                obs, rewards_t, terminations, truncations, _ = envs.step(actions_t)
                dones_t = terminations | truncations

                states.append(states_t)
                taken_actions.append(actions_t)
                values.append(np.atleast_1d(values_t).reshape(-1))
                rewards.append(rewards_t)
                dones.append(dones_t.astype(np.float32))

                for i in range(args.num_envs):
                    if dones_t[i]:
                        for _ in range(FRAME_STACK_SIZE):
                            frame_stacks[i].add_frame(obs[i])
                    else:
                        frame_stacks[i].add_frame(obs[i])

            # Bootstrap values
            states_last = [frame_stacks[i].get_state() for i in range(args.num_envs)]
            _, last_values = agent.predict(states_last, use_old_policy=True, greedy=False)
            last_values = np.atleast_1d(last_values).reshape(-1)

            # GAE
            rewards_arr = np.array(rewards)
            values_arr = np.array(values)
            dones_arr = np.array(dones)
            advantages = compute_gae(
                rewards_arr,
                values_arr,
                last_values,
                dones_arr,
                args.discount_factor,
                args.gae_lambda,
            )
            returns = advantages + values_arr
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Flatten
            T, N = args.horizon, args.num_envs
            states_flat = np.array(states).reshape(-1, *INPUT_SHAPE)
            actions_flat = np.array(taken_actions).reshape(-1, num_actions)
            returns_flat = returns.flatten()
            advantages_flat = advantages.flatten()

            # PPO update
            agent.update_old_policy()
            for _ in range(args.num_epochs):
                indices = np.arange(T * N)
                np.random.shuffle(indices)
                for start in range(0, T * N, args.batch_size):
                    mb_idx = indices[start : start + args.batch_size]

                    if agent.step_idx % args.eval_interval == 0:
                        pbar.write("[INFO] Running evaluation...")
                        avg_reward, value_error = evaluate(
                            agent, FRAME_STACK_SIZE, step_idx=agent.step_idx,
                            max_steps=args.eval_max_steps,
                        )
                        agent.write_to_summary("eval_reward", avg_reward)
                        agent.write_to_summary("eval_value_error", value_error)
                        pbar.write(f"  eval_reward={avg_reward:.1f}, value_error={value_error:.4f}")

                    if agent.step_idx % args.save_interval == 0:
                        agent.save()
                        pbar.write(f"[INFO] Model saved at step {agent.step_idx}")

                    agent.train(
                        states_flat[mb_idx],
                        actions_flat[mb_idx],
                        returns_flat[mb_idx],
                        advantages_flat[mb_idx],
                        learning_rate=lr_scheduler,
                    )
                    pbar.update(1)
                    pbar.set_postfix(step=agent.step_idx)

                    if max_train_steps is not None and agent.step_idx >= max_train_steps:
                        break
                if max_train_steps is not None and agent.step_idx >= max_train_steps:
                    break
            if max_train_steps is not None and agent.step_idx >= max_train_steps:
                break
    except KeyboardInterrupt:
        agent.save()
        pbar.write("[INFO] Training interrupted, model saved")
    finally:
        pbar.close()

    envs.close()


if __name__ == "__main__":
    train()
