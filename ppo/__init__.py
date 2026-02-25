"""
ppo 包初始化模块
================
本包实现了 PPO (Proximal Policy Optimization) 算法的核心组件。

导出的公共接口：
    - PPOAgent      : PPO 训练代理，封装了网络、优化器、训练和推理逻辑
    - FrameStack     : 帧堆叠工具，将连续多帧图像堆叠为单个状态
    - preprocess_frame : 图像预处理函数，将 RGB 帧转为灰度并归一化
    - compute_gae    : GAE (Generalized Advantage Estimation) 计算函数
"""

from .agent import PPOAgent
from .utils import FrameStack, preprocess_frame, compute_gae

__all__ = ["PPOAgent", "FrameStack", "preprocess_frame", "compute_gae"]
