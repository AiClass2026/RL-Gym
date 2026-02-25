"""
PPO 工具函数模块
================
提供强化学习训练中常用的辅助工具：

1. preprocess_frame  — 图像预处理（裁剪、灰度化、归一化）
2. FrameStack        — 帧堆叠器，将连续多帧组合为一个状态
3. compute_gae       — GAE (Generalized Advantage Estimation) 优势估计

为什么需要这些工具？
- 原始环境输出 96×96 RGB 图像，信息冗余且维度高。预处理可减小输入规模、加速训练。
- 单帧图像无法反映运动方向和速度，帧堆叠让智能体"看到"时间维度的变化。
- GAE 在偏差 (bias) 和方差 (variance) 之间取得平衡，比朴素的优势估计更稳定。
"""

from collections import deque
import numpy as np


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    将环境输出的 RGB 帧预处理为神经网络输入格式。

    处理步骤：
        1. 裁剪：去掉底部 12 行（仪表盘区域）和左右各 6 列 → 84×84
        2. 灰度化：使用 ITU-R BT.601 标准权重 (0.299R + 0.587G + 0.114B)
        3. 归一化：将像素值从 [0, 255] 映射到 [-1, 1]

    Args:
        frame: 原始 RGB 帧，形状为 (96, 96, 3)，像素值 [0, 255]

    Returns:
        处理后的灰度帧，形状为 (84, 84)，像素值 [-1, 1]，dtype=float32
    """
    # 裁剪：去掉底部仪表盘 (12行) 和左右边框 (各6列)，得到 84×84 的赛道区域
    frame = frame[:-12, 6:-6]

    # 灰度化：RGB 三通道按人眼感知权重加权求和
    frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])

    # 归一化到 [-1, 1]：先除以 255 得到 [0,1]，再线性变换到 [-1,1]
    # 使用 [-1,1] 而非 [0,1] 是因为零中心化的输入有助于神经网络收敛
    frame = frame / 255.0 * 2 - 1

    return frame.astype(np.float32)


class FrameStack:
    """
    帧堆叠器：维护一个固定大小的滑动窗口，将最近 N 帧堆叠为一个状态。

    在 Atari / 赛车等环境中，单帧图像无法提供运动信息（如速度、方向）。
    通过堆叠连续多帧，神经网络可以从帧间差异中推断出动态信息。

    示例：
        stack_size=4 时，状态形状为 (84, 84, 4)，即 4 个灰度帧沿最后一个轴堆叠。

    Attributes:
        frame_stack: 双端队列，存储最近 stack_size 帧
        preprocess_fn: 可选的预处理函数，在帧入栈前自动调用
    """

    def __init__(self, initial_frame: np.ndarray, stack_size: int = 4, preprocess_fn=None):
        """
        初始化帧堆叠器。

        Args:
            initial_frame: 环境重置后的第一帧原始图像
            stack_size: 堆叠帧数（默认 4）
            preprocess_fn: 预处理函数，如 preprocess_frame；为 None 则不预处理
        """
        self.frame_stack = deque(maxlen=stack_size)
        # 用初始帧填满整个队列，确保从第一步起就有完整的堆叠状态
        initial_frame = preprocess_fn(initial_frame) if preprocess_fn else initial_frame
        for _ in range(stack_size):
            self.frame_stack.append(initial_frame)
        self.preprocess_fn = preprocess_fn

    def add_frame(self, frame: np.ndarray) -> None:
        """
        向堆叠中添加新帧（自动丢弃最旧的帧）。

        Args:
            frame: 新的原始帧图像
        """
        self.frame_stack.append(self.preprocess_fn(frame) if self.preprocess_fn else frame)

    def get_state(self) -> np.ndarray:
        """
        获取当前堆叠状态。

        Returns:
            堆叠后的状态数组，形状为 (H, W, stack_size)
            例如 stack_size=4 时返回 (84, 84, 4)
        """
        # 沿最后一个轴堆叠，与 CNN 输入格式对应 (H, W, C)
        return np.stack(self.frame_stack, axis=-1)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap_values: np.ndarray,
    terminals: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """
    计算 GAE (Generalized Advantage Estimation) 优势估计。

    GAE 的核心思想：
        - TD 残差 (δ_t) = r_t + γ * V(s_{t+1}) - V(s_t)
        - 优势函数 A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
        - λ=0 时退化为 1-step TD（低方差、高偏差）
        - λ=1 时退化为 Monte Carlo（高方差、低偏差）
        - 通常 λ=0.95 提供偏差和方差的良好平衡

    Args:
        rewards: 每步奖励，形状 [T, N]，T=时间步数，N=并行环境数
        values: 每步状态价值 V(s)，形状 [T, N]
        bootstrap_values: 最后一步的下一状态价值 V(s_{T+1})，形状 [N]
        terminals: 是否终止的标记，形状 [T, N]（终止=1.0，否则=0.0）
        gamma: 折扣因子 γ，控制对未来奖励的衰减（通常 0.99）
        lam: GAE 的 λ 参数，控制偏差-方差权衡（通常 0.95）

    Returns:
        advantages: 每步的优势估计，形状 [T, N]
    """
    # 在 values 末尾拼接 bootstrap_values，方便索引 values[i+1]
    values = np.vstack((values, [bootstrap_values]))

    # 第一步：计算每步的 TD 残差 δ_t = r_t + γ(1-done_t) * V(s_{t+1}) - V(s_t)
    # 如果 episode 终止 (done=1)，则不用下一步的价值（未来不存在）
    deltas = []
    for i in reversed(range(len(rewards))):
        V = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))

    # 第二步：从后往前累积计算 GAE
    # A_t = δ_t + γλ(1-done_t) * A_{t+1}
    # 从最后一步开始，逐步向前传播优势信号
    A = deltas[-1, :]
    advantages = [A]
    for i in reversed(range(len(deltas) - 1)):
        A = deltas[i] + (1.0 - terminals[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)

    return np.array(list(advantages))
