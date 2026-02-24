"""PPO 工具函数：预处理、FrameStack、GAE 计算"""
from collections import deque
import numpy as np


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """将 96x96x3 RGB 帧预处理为 84x84 灰度并归一化到 [-1, 1]"""
    frame = frame[:-12, 6:-6]  # crop to 84x84
    frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])  # grayscale
    frame = frame / 255.0 * 2 - 1  # normalize to [-1, 1]
    return frame.astype(np.float32)


class FrameStack:
    """维护多帧堆叠的状态"""

    def __init__(self, initial_frame: np.ndarray, stack_size: int = 4, preprocess_fn=None):
        self.frame_stack = deque(maxlen=stack_size)
        initial_frame = preprocess_fn(initial_frame) if preprocess_fn else initial_frame
        for _ in range(stack_size):
            self.frame_stack.append(initial_frame)
        self.preprocess_fn = preprocess_fn

    def add_frame(self, frame: np.ndarray) -> None:
        self.frame_stack.append(self.preprocess_fn(frame) if self.preprocess_fn else frame)

    def get_state(self) -> np.ndarray:
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
    计算 GAE (Generalized Advantage Estimation)
    rewards: [T, N], values: [T, N], bootstrap_values: [N], terminals: [T, N]
    返回 advantages: [T, N]
    """
    values = np.vstack((values, [bootstrap_values]))

    # Compute delta
    deltas = []
    for i in reversed(range(len(rewards))):
        V = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))

    # Compute GAE
    A = deltas[-1, :]
    advantages = [A]
    for i in reversed(range(len(deltas) - 1)):
        A = deltas[i] + (1.0 - terminals[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)

    return np.array(list(advantages))
