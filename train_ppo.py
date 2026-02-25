"""
CarRacing-v3 PPO 训练主脚本
============================
本脚本实现了完整的 PPO 训练流程：

    1. 解析命令行超参数
    2. 创建并行向量化环境（AsyncVectorEnv）
    3. 执行"采样 → 计算 GAE → PPO 更新"的训练循环
    4. 定期评估智能体性能并录制视频
    5. 定期保存模型检查点

训练流程示意：

    ┌─────────────────────────────────────────────┐
    │  Rollout（数据采集）                          │
    │  在 num_envs 个并行环境中各运行 horizon 步      │
    │  收集 states, actions, rewards, log_probs    │
    └──────────────────┬──────────────────────────┘
                       ▼
    ┌─────────────────────────────────────────────┐
    │  计算 GAE 优势估计 + 折扣回报                  │
    └──────────────────┬──────────────────────────┘
                       ▼
    ┌─────────────────────────────────────────────┐
    │  PPO 更新（num_epochs 轮 mini-batch 训练）    │
    │  每个 mini-batch 执行一次策略梯度更新           │
    └──────────────────┬──────────────────────────┘
                       ▼
              回到 Rollout 继续...

用法：
    python train_ppo.py
    python train_ppo.py --num_envs 32 --max_train_steps 100000 --device cuda:0
"""

import os
import argparse
from datetime import datetime

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm

from ppo import PPOAgent, FrameStack, preprocess_frame, compute_gae

# ============================================================
# 全局常量
# ============================================================
ENV_NAME = "CarRacing-v3"       # Gymnasium 环境 ID
FRAME_STACK_SIZE = 4            # 堆叠帧数：4 帧提供速度和方向信息
INPUT_SHAPE = (84, 84, FRAME_STACK_SIZE)  # 神经网络输入形状 (高, 宽, 通道)


def parse_args():
    """
    解析命令行参数，定义所有可调超参数。

    Returns:
        args: 解析后的参数命名空间对象
    """
    parser = argparse.ArgumentParser(
        description="PPO 训练 CarRacing-v3"
    )
    # ---- 学习相关 ----
    parser.add_argument("--initial_lr", type=float, default=3e-4, help="初始学习率")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="折扣因子 γ")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE 的 λ 参数")
    parser.add_argument("--ppo_epsilon", type=float, default=0.2, help="PPO 裁剪参数 ε")
    parser.add_argument("--value_scale", type=float, default=0.5, help="价值损失缩放系数")
    parser.add_argument("--entropy_scale", type=float, default=0.01, help="熵损失缩放系数")

    # ---- 采样与更新 ----
    parser.add_argument("--horizon", type=int, default=64, help="每次 PPO 更新前，每个 env 采样步数")
    parser.add_argument("--num_epochs", type=int, default=10, help="每次 rollout 后的 PPO 更新轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="mini-batch 大小")
    parser.add_argument("--num_envs", type=int, default=64, help="并行环境数量")

    # ---- 运行控制 ----
    parser.add_argument("--model_name", type=str, default="CarRacing-v3", help="模型名称，用于 run 目录后缀")
    parser.add_argument("--save_interval", type=int, default=10000, help="保存 checkpoint 的间隔（训练步数）")
    parser.add_argument("--eval_interval", type=int, default=2000, help="评估的间隔（训练步数）")
    parser.add_argument("--eval_max_steps", type=int, default=None, help="每次评估的最大环境步数（默认不限制）")
    parser.add_argument("--max_train_steps", type=int, default=280000, help="最大训练步数，达到后自动停止")
    parser.add_argument("--device", type=str, default=None, help="设备，如 cuda:0 / cuda / cpu；默认自动选 cuda")
    parser.add_argument("--run_id", type=str, default=None, help="运行唯一标识，多进程时避免目录冲突；默认用 PID")
    return parser.parse_args()


def make_env():
    """
    创建单个 CarRacing-v3 环境实例（工厂函数）。

    Returns:
        gym.Env: 配置好的赛车环境
    """
    return gym.make(
        ENV_NAME,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )


def evaluate(agent, frame_stack_size, step_idx=None, max_steps=None):
    """
    评估智能体性能：运行一个完整 episode，计算总奖励和价值预测误差。

    评估流程：
        1. 创建独立的测试环境（带视频录制）
        2. 用贪心策略（不加噪声）运行一个 episode
        3. 收集奖励和 Critic 的价值预测
        4. 通过对比预测值和实际回报来计算 value error

    value error 的意义：
        衡量 Critic 预测的准确性。如果 value error 持续下降，
        说明 Critic 越来越能准确评估状态的好坏。

    Args:
        agent: PPOAgent 实例
        frame_stack_size: 帧堆叠数量
        step_idx: 当前训练步数（用于视频文件命名）
        max_steps: 评估的最大步数限制（None 表示不限制）

    Returns:
        total_reward: 整个 episode 的累计奖励
        value_error: Critic 预测值与实际回报的均方误差
    """
    # 创建独立的测试环境（与训练环境隔离）
    test_env = gym.make(
        ENV_NAME,
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )
    # 用 RecordVideo 录制评估视频，文件名包含训练步数便于追踪进展
    name_prefix = f"step_{step_idx}" if step_idx is not None else "eval"
    test_env = RecordVideo(
        test_env,
        video_folder=agent.video_dir,
        episode_trigger=lambda x: True,
        name_prefix=name_prefix,
    )

    # 重置环境并初始化帧堆叠
    obs, _ = test_env.reset(seed=0)
    frame_stack = FrameStack(obs, stack_size=frame_stack_size, preprocess_fn=preprocess_frame)

    total_reward = 0.0
    values_list = []    # 收集每步的 Critic 价值预测
    rewards_list = []   # 收集每步的实际奖励
    dones_list = []     # 收集每步的终止标记

    step_count = 0
    while True:
        state = frame_stack.get_state()
        # 贪心模式：直接使用策略均值作为动作（无探索噪声）
        action, value, _, _ = agent.predict([state], greedy=True)
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

    # ---- 计算 value error ----
    # 从最后一步向前计算折扣回报 R_t = r_t + γ * R_{t+1}
    # 然后与 Critic 的预测值对比
    last_value = agent.predict([frame_stack.get_state()], greedy=True)[1][0]
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
    """
    PPO 训练主函数。

    整体流程：
        1. 解析参数、创建环境和 Agent
        2. 进入训练循环：
           a. Rollout：在并行环境中采集经验数据
           b. GAE：计算优势估计和折扣回报
           c. PPO Update：用 mini-batch 多轮更新策略
        3. 定期评估和保存检查点
    """
    args = parse_args()

    # 学习率指数衰减调度器：每 10000 步衰减为 0.85 倍
    def lr_scheduler(step_idx):
        return args.initial_lr * 0.85 ** (step_idx // 10000)

    # ============================================================
    # 第一步：获取动作空间信息
    # ============================================================
    # 临时创建环境以查询动作空间的维度和范围
    test_env = gym.make(
        ENV_NAME,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )
    action_space = test_env.action_space
    num_actions = int(action_space.shape[0])    # 动作维度数 = 3
    action_min = np.array(action_space.low, dtype=np.float32)   # [-1, 0, 0]
    action_max = np.array(action_space.high, dtype=np.float32)  # [+1, 1, 1]
    test_env.close()

    # ============================================================
    # 第二步：创建运行目录和 Agent
    # ============================================================
    # 目录命名格式：{时间戳}_{运行ID}_{模型名}，确保每次运行不冲突
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id if args.run_id is not None else str(os.getpid())
    run_dir = os.path.join("./runs", f"{timestamp}_{run_id}_{args.model_name}")
    print(f"[INFO] Run directory: {run_dir}")

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

    # ============================================================
    # 第三步：创建并行环境
    # ============================================================
    # AsyncVectorEnv: 每个环境在独立子进程中运行，充分利用多核 CPU
    envs = gym.make_vec(
        ENV_NAME,
        num_envs=args.num_envs,
        vectorization_mode="async",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )

    # ============================================================
    # 第四步：初始化环境和帧堆叠
    # ============================================================
    obs, _ = envs.reset()
    # 为每个并行环境创建独立的帧堆叠器
    frame_stacks = [
        FrameStack(obs[i], stack_size=FRAME_STACK_SIZE, preprocess_fn=preprocess_frame)
        for i in range(args.num_envs)
    ]

    max_train_steps = args.max_train_steps
    print("[INFO] Training loop started" + (f" (max {max_train_steps} steps)" if max_train_steps else ""))

    # 进度条：实时显示训练进度和速度
    pbar = tqdm(
        total=max_train_steps,
        unit="step",
        unit_scale=False,
        desc="Training",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    # ============================================================
    # 第五步：训练主循环
    # ============================================================
    try:
        while True:
            # 检查是否达到最大训练步数
            if max_train_steps is not None and agent.step_idx >= max_train_steps:
                break

            # ---- 阶段 A：Rollout（经验采集）----
            # 在所有并行环境中各运行 horizon 步，收集训练数据
            states, taken_actions, log_probs, values, rewards, dones = [], [], [], [], [], []

            for _ in range(args.horizon):
                # 获取所有环境的当前堆叠状态
                states_t = [frame_stacks[i].get_state() for i in range(args.num_envs)]

                # 用当前策略采样动作（随机模式，带探索噪声）
                actions_clipped, values_t, actions_for_train, log_probs_t = agent.predict(states_t)

                # 在所有环境中同时执行动作
                obs, rewards_t, terminations, truncations, _ = envs.step(actions_clipped)
                dones_t = terminations | truncations

                # 存储这一步的数据
                states.append(states_t)
                taken_actions.append(actions_for_train) # 存储裁剪后的动作（与环境执行的一致）
                log_probs.append(log_probs_t)           # 存储采样时的 log 概率
                values.append(np.atleast_1d(values_t).reshape(-1))
                rewards.append(rewards_t)
                dones.append(dones_t.astype(np.float32))

                # 更新帧堆叠
                for i in range(args.num_envs):
                    if dones_t[i]:
                        # episode 结束时，用重置后的观测填满整个帧堆叠
                        # 避免新 episode 的状态混入旧 episode 的帧
                        for _ in range(FRAME_STACK_SIZE):
                            frame_stacks[i].add_frame(obs[i])
                    else:
                        frame_stacks[i].add_frame(obs[i])

            # ---- 阶段 B：Bootstrap + GAE 计算 ----
            # 用最后状态的价值估计作为 bootstrap（自举值）
            # 因为 rollout 并不总是在 episode 结束时停止，需要估计"未来还能获得多少奖励"
            states_last = [frame_stacks[i].get_state() for i in range(args.num_envs)]
            _, last_values, _, _ = agent.predict(states_last)
            last_values = np.atleast_1d(last_values).reshape(-1)

            # 计算 GAE 优势估计
            rewards_arr = np.array(rewards)     # [T, N]
            values_arr = np.array(values)       # [T, N]
            dones_arr = np.array(dones)         # [T, N]
            advantages = compute_gae(
                rewards_arr,
                values_arr,
                last_values,
                dones_arr,
                args.discount_factor,
                args.gae_lambda,
            )

            # 回报 = 优势 + 价值基线
            returns = advantages + values_arr

            # 优势标准化：减均值除标准差，使优势分布集中在 0 附近
            # 这有助于稳定 PPO 的策略梯度更新
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ---- 阶段 C：数据展平 ----
            # 将 [T, N, ...] 形状的数据展平为 [T*N, ...] 便于 mini-batch 采样
            T, N = args.horizon, args.num_envs
            states_flat = np.array(states).reshape(-1, *INPUT_SHAPE)       # [T*N, 84, 84, 4]
            actions_flat = np.array(taken_actions).reshape(-1, num_actions) # [T*N, 3]
            log_probs_flat = np.array(log_probs).flatten()                 # [T*N]
            returns_flat = returns.flatten()                                # [T*N]
            advantages_flat = advantages.flatten()                         # [T*N]

            # ---- 阶段 D：PPO 多轮更新 ----
            # 对同一批 rollout 数据训练 num_epochs 轮
            # 每轮随机打乱数据，按 batch_size 切分 mini-batch
            for _ in range(args.num_epochs):
                indices = np.arange(T * N)
                np.random.shuffle(indices)
                for start in range(0, T * N, args.batch_size):
                    mb_idx = indices[start : start + args.batch_size]

                    # 定期评估
                    if agent.step_idx % args.eval_interval == 0:
                        pbar.write("[INFO] Running evaluation...")
                        avg_reward, value_error = evaluate(
                            agent, FRAME_STACK_SIZE, step_idx=agent.step_idx,
                            max_steps=args.eval_max_steps,
                        )
                        agent.write_to_summary("eval_reward", avg_reward)
                        agent.write_to_summary("eval_value_error", value_error)
                        pbar.write(f"  eval_reward={avg_reward:.1f}, value_error={value_error:.4f}")

                    # 定期保存检查点
                    if agent.step_idx % args.save_interval == 0:
                        agent.save()
                        pbar.write(f"[INFO] Model saved at step {agent.step_idx}")

                    # 执行一步 PPO 参数更新
                    agent.train(
                        states_flat[mb_idx],
                        actions_flat[mb_idx],
                        log_probs_flat[mb_idx],
                        returns_flat[mb_idx],
                        advantages_flat[mb_idx],
                        learning_rate=lr_scheduler,
                    )
                    pbar.update(1)
                    pbar.set_postfix(step=agent.step_idx)

                    # 检查是否达到最大步数（三层循环中都需要检查）
                    if max_train_steps is not None and agent.step_idx >= max_train_steps:
                        break
                if max_train_steps is not None and agent.step_idx >= max_train_steps:
                    break
            if max_train_steps is not None and agent.step_idx >= max_train_steps:
                break
    except KeyboardInterrupt:
        # Ctrl+C 中断时自动保存模型，避免丢失训练进度
        agent.save()
        pbar.write("[INFO] Training interrupted, model saved")
    finally:
        pbar.close()

    envs.close()


if __name__ == "__main__":
    train()
