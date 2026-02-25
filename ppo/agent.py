"""
PPO Agent 模块：Actor-Critic 网络 + PPO 训练代理
===================================================
本模块实现了 PPO 算法的两个核心组件：

1. ActorCritic 网络
   - 共享的 CNN 特征提取器
   - Actor 头：输出连续动作的高斯分布参数（均值 + 标准差）
   - Critic 头：输出状态价值 V(s)

2. PPOAgent 训练代理
   - 封装推理 (predict)、训练 (train)、保存 (save) 等操作
   - Rollout 阶段保存 log_prob，训练时直接使用（标准 PPO 写法）
   - 集成 TensorBoard 日志记录

网络架构示意：
    输入 (84,84,4) → Conv1(16, 8×8, s4) → Conv2(32, 3×3, s2) → FC(256)
                                                                    ├→ Actor: mean + log_std → 动作分布
                                                                    └→ Critic: V(s) → 状态价值
"""

import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


class ActorCritic(nn.Module):
    """
    Actor-Critic 网络：共享 CNN 特征 + 策略头 + 价值头。

    PPO 使用 Actor-Critic 架构的原因：
    - Actor（策略网络）决定在每个状态下采取什么动作
    - Critic（价值网络）评估每个状态的好坏，为 Actor 提供训练信号
    - 共享底层 CNN 特征可以减少参数量、加速训练

    Attributes:
        num_actions: 动作空间维度（CarRacing 中为 3：转向、油门、刹车）
        action_min: 每个动作维度的最小值
        action_max: 每个动作维度的最大值
    """

    def __init__(self, input_shape, num_actions, action_min, action_max):
        """
        初始化 Actor-Critic 网络。

        Args:
            input_shape: 输入图像形状，如 (84, 84, 4)，最后一维是堆叠帧数
            num_actions: 连续动作空间的维度数（CarRacing 中为 3）
            action_min: 动作各维度最小值数组，如 [-1, 0, 0]
            action_max: 动作各维度最大值数组，如 [+1, 1, 1]
        """
        super().__init__()
        self.num_actions = num_actions
        self.action_min = torch.tensor(action_min, dtype=torch.float32)
        self.action_max = torch.tensor(action_max, dtype=torch.float32)
        # register_buffer 使张量跟随模型移动到 GPU/CPU，但不作为可训练参数
        self.register_buffer("_action_min", self.action_min)
        self.register_buffer("_action_max", self.action_max)

        # ---- CNN 特征提取器 ----
        # 输入通道数 = 堆叠帧数（如 4），使用 channels-last 格式转换
        # Conv1: 16 个 8×8 卷积核，步长 4 → 大幅降低空间分辨率，提取低级特征
        self.conv1 = nn.Conv2d(input_shape[-1], 16, kernel_size=8, stride=4)
        # Conv2: 32 个 3×3 卷积核，步长 2 → 进一步提取高级特征
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        # 计算 CNN 输出的展平维度（用于连接全连接层）
        # 公式：output_size = (input_size - kernel_size) / stride + 1
        def conv_out_size(size, kernel, stride):
            return (size - kernel) // stride + 1

        h = conv_out_size(conv_out_size(input_shape[0], 8, 4), 3, 2)  # 84→20→9
        w = conv_out_size(conv_out_size(input_shape[1], 8, 4), 3, 2)  # 84→20→9
        self.feature_dim = 32 * h * w  # 32 通道 × 9 × 9 = 2592

        # 全连接层：将 CNN 特征映射到 256 维的隐藏表示
        self.fc = nn.Linear(self.feature_dim, 256)

        # ---- Actor 头（策略网络）----
        # 输出连续动作的高斯分布参数
        # actor_mean: 均值向量，经 tanh 映射到 [-1, 1] 后再缩放到动作范围
        self.actor_mean = nn.Linear(256, num_actions)
        # actor_logstd: 对数标准差，初始化为 log(0.4) ≈ -0.916
        # 使用 log 空间确保 std 始终为正；初始值 0.4 提供适度的探索
        self.actor_logstd = nn.Parameter(torch.full((num_actions,), np.log(0.4)))

        # ---- Critic 头（价值网络）----
        # 输出标量状态价值 V(s)
        self.critic = nn.Linear(256, 1)

    def _forward_features(self, x):
        """
        CNN 特征提取前向传播。

        Args:
            x: 输入张量，形状 (N, H, W, C) 或 (N, C, H, W)
               其中 N=batch, H=高, W=宽, C=通道数（堆叠帧数）

        Returns:
            特征向量，形状 (N, 256)
        """
        # PyTorch Conv2d 要求 channels-first 格式 (N, C, H, W)
        # 如果输入是 channels-last (N, H, W, C)，需要转置
        if x.shape[-1] == 4:
            x = x.permute(0, 3, 1, 2).contiguous()

        # LeakyReLU 激活函数：比 ReLU 对负值有微小梯度，避免"神经元死亡"
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))

        # 将 (N, 32, 9, 9) 展平为 (N, 2592)
        x = x.reshape(x.size(0), -1)

        x = F.leaky_relu(self.fc(x))
        return x

    def forward(self, x):
        """
        完整前向传播：输入状态 → 输出动作分布参数和状态价值。

        Args:
            x: 输入状态张量，形状 (N, 84, 84, 4)

        Returns:
            mean: 动作均值，已缩放到 [action_min, action_max]，形状 (N, num_actions)
            std: 动作标准差，形状 (N, num_actions)
            value: 状态价值 V(s)，形状 (N,)
        """
        features = self._forward_features(x)

        # Actor：计算动作均值
        # tanh 将输出压缩到 [-1, 1]，再线性映射到 [action_min, action_max]
        mean_raw = torch.tanh(self.actor_mean(features))
        action_min = self._action_min.to(x.device)
        action_max = self._action_max.to(x.device)
        # 线性映射公式：mean = min + (tanh_output + 1) / 2 * (max - min)
        mean = action_min + (mean_raw + 1) / 2 * (action_max - action_min)

        # 标准差：对 log_std 取 exp，并扩展到与 mean 相同形状
        # 关键：对 log_std 施加下界 log(0.01)，防止"熵崩溃"（entropy collapse）。
        # 如果不加限制，log_std 可能被优化到极小值（如 -10），使 std ≈ 0，
        # 策略退化为确定性策略，丧失探索能力，一旦锁定在差的动作模式上就无法恢复。
        # 这是导致"相同超参数多次运行结果差异巨大"的主要原因之一。
        clamped_logstd = torch.clamp(self.actor_logstd, min=np.log(0.01))
        std = torch.exp(clamped_logstd).expand_as(mean)

        # Critic：状态价值（squeeze 去掉最后的维度 1）
        value = self.critic(features).squeeze(-1)

        return mean, std, value

    def get_action_and_value(self, x, action=None):
        """
        采样动作并计算 log 概率、熵和价值（用于训练）。

        PPO 训练时需要：
        - log_prob: 当前策略对动作的对数概率（用于计算重要性采样比率）
        - entropy: 策略的熵（鼓励探索，防止过早收敛到确定性策略）
        - value: 状态价值（用于计算优势函数和价值损失）

        Args:
            x: 输入状态张量，形状 (N, 84, 84, 4)
            action: 指定动作（训练时传入已采样的动作）；为 None 则重新采样

        Returns:
            action: 采样的动作，形状 (N, num_actions)
            log_prob: 动作的对数概率，形状 (N,)
            entropy: 策略熵，形状 (N,)
            value: 状态价值，形状 (N,)
        """
        mean, std, value = self.forward(x)

        # 构建多维正态分布 N(mean, std)
        dist = Normal(mean, std)

        if action is None:
            # 推理时：从分布中随机采样动作（探索）
            action = dist.sample()

        # 计算每个动作维度的 log 概率，然后对维度求和
        # 对数概率求和等价于联合概率的对数（假设各维度独立）
        log_prob = dist.log_prob(action).sum(dim=-1)

        # 熵衡量策略的不确定性，值越大说明策略越"随机"，有利于探索
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value


class PPOAgent:
    """
    PPO 训练代理：封装完整的 PPO 训练流程。

    PPO 的核心思想：
        限制每次策略更新的幅度（通过裁剪比率），防止策略突变导致训练崩溃。

    PPO 损失函数由三部分组成：
        L = L_policy + c1 * L_value + c2 * L_entropy

        1. L_policy（策略损失）：裁剪后的策略梯度，使策略不会偏离太远
        2. L_value（价值损失）：Critic 预测值与实际回报的 MSE
        3. L_entropy（熵损失）：鼓励探索的正则项

    标准 PPO 实现：Rollout 阶段用当前策略采样并保存 log_prob，
    训练时直接使用保存的 log_prob 计算重要性采样比率，无需维护旧策略网络副本。

    Attributes:
        epsilon: PPO 裁剪范围 ε（默认 0.2，即限制比率在 [0.8, 1.2] 之间）
        value_scale: 价值损失的缩放系数 c1
        entropy_scale: 熵损失的缩放系数 c2
        step_idx: 当前训练步数计数器
    """

    def __init__(
        self,
        input_shape,
        num_actions,
        action_min,
        action_max,
        epsilon=0.2,
        value_scale=0.5,
        entropy_scale=0.01,
        run_dir="./runs/default",
        device=None,
    ):
        """
        初始化 PPO 代理。

        Args:
            input_shape: 输入状态形状，如 (84, 84, 4)
            num_actions: 动作空间维度
            action_min: 动作各维度最小值
            action_max: 动作各维度最大值
            epsilon: PPO 裁剪参数 ε，限制策略更新幅度（默认 0.2）
            value_scale: 价值损失缩放系数（默认 0.5）
            entropy_scale: 熵损失缩放系数（默认 0.01）
            run_dir: 本次运行的输出目录（检查点、日志、视频）
            device: 计算设备，如 "cuda:0" / "cpu"；None 则自动选择
        """
        self.epsilon = epsilon
        self.value_scale = value_scale
        self.entropy_scale = entropy_scale
        self.step_idx = 0

        # 创建输出目录结构
        self.run_dir = run_dir
        self.ckpt_dir = os.path.join(run_dir, "ckpt")     # 模型检查点
        self.log_dir = os.path.join(run_dir, "logs")       # TensorBoard 日志
        self.video_dir = os.path.join(run_dir, "videos")   # 评估视频

        for d in [self.ckpt_dir, self.log_dir, self.video_dir]:
            os.makedirs(d, exist_ok=True)

        # 设备选择：优先使用 GPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 创建策略网络
        self.policy = ActorCritic(input_shape, num_actions, action_min, action_max).to(self.device)

        # Adam 优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

        # TensorBoard 日志记录器
        self.writer = SummaryWriter(self.log_dir)

    def _to_tensor(self, x):
        """
        将 numpy 数组转换为 PyTorch 张量并移动到对应设备。

        Args:
            x: numpy 数组或已有张量

        Returns:
            float32 类型的 PyTorch 张量，位于 self.device 上
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        return x

    def predict(self, states, greedy=False):
        """
        根据当前状态预测动作。

        有两种模式：
        - 贪心模式 (greedy=True)：直接使用均值作为动作（评估时使用）
        - 随机模式 (greedy=False)：从高斯分布中采样（训练时使用，增加探索）

        Args:
            states: 状态列表或数组，每个元素形状为 (84, 84, 4)
            greedy: 是否使用贪心策略（不加噪声）

        Returns:
            action_clipped: 裁剪到合法范围的动作，形状 (N, num_actions)
            value: 状态价值估计，形状 (N,)
            action_for_train: 用于训练的动作（= action_clipped，greedy 模式下为 None）
            log_prob: 对 action_clipped 的对数概率（greedy 模式下为 None）
        """
        x = self._to_tensor(np.array(states))

        # 推理阶段不需要计算梯度
        with torch.no_grad():
            if greedy:
                # 贪心模式：直接用均值作为动作，不添加随机噪声
                mean, _, value = self.policy(x)
                action_clipped = torch.clamp(mean, self.policy._action_min, self.policy._action_max)
                return action_clipped.cpu().numpy(), value.cpu().numpy(), None, None
            else:
                # 随机模式：从高斯分布中采样，用于探索
                mean, std, value = self.policy(x)
                dist = Normal(mean, std)
                action_raw = dist.sample()
                action_clipped = torch.clamp(action_raw, self.policy._action_min, self.policy._action_max)
                # log_prob 基于 clipped 动作，与环境实际执行的动作一致
                log_prob = dist.log_prob(action_clipped).sum(dim=-1)
                return (
                    action_clipped.cpu().numpy(),
                    value.cpu().numpy(),
                    action_clipped.cpu().numpy(),
                    log_prob.cpu().numpy(),
                )

    def train(self, states, taken_actions, log_prob_old, returns, advantages, learning_rate):
        """
        执行一步 PPO 参数更新。

        PPO 更新的核心步骤：
        1. 用当前策略重新计算对已采样动作的 log 概率
        2. 计算重要性采样比率 r = exp(log_prob_new - log_prob_old)
        3. 用裁剪函数限制比率范围，防止策略偏离过大
        4. 组合策略损失、价值损失和熵损失
        5. 反向传播并更新参数

        Args:
            states: 状态 batch，形状 (B, 84, 84, 4)
            taken_actions: 已执行的动作，形状 (B, num_actions)
            log_prob_old: 旧策略下动作的 log 概率，形状 (B,)
            returns: 折扣回报（GAE 计算得到），形状 (B,)
            advantages: 优势估计值（已标准化），形状 (B,)
            learning_rate: 学习率，可以是浮点数或接受 step_idx 的可调用对象

        Returns:
            (total_loss, policy_loss, value_loss, entropy_loss) 四元组
        """
        # 将所有输入转换为张量
        states = self._to_tensor(states)
        taken_actions = self._to_tensor(taken_actions)
        log_prob_old = self._to_tensor(log_prob_old)
        returns = self._to_tensor(returns)
        advantages = self._to_tensor(advantages)

        # 动态调整学习率（支持学习率调度器）
        for p in self.optimizer.param_groups:
            p["lr"] = learning_rate(self.step_idx) if callable(learning_rate) else learning_rate

        # 用当前策略重新评估之前采样的动作
        _, log_prob_new, entropy, value = self.policy.get_action_and_value(states, taken_actions)

        # ---- 策略损失 (Policy Loss) ----
        # 重要性采样比率：r(θ) = π_new(a|s) / π_old(a|s) = exp(logπ_new - logπ_old)
        ratio = torch.exp(log_prob_new - log_prob_old)

        # 未裁剪的策略目标
        policy_loss_1 = ratio * advantages
        # 裁剪后的策略目标：将比率限制在 [1-ε, 1+ε] 范围内
        policy_loss_2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
        # 取两者的较小值（悲观估计），确保训练稳定
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # ---- 价值损失 (Value Loss) ----
        # Critic 预测值与实际回报的均方误差
        value_loss = F.mse_loss(value, returns) * self.value_scale

        # ---- 熵损失 (Entropy Loss) ----
        # 负号因为我们要最大化熵（鼓励探索），而优化器默认做最小化
        entropy_loss = -entropy.mean() * self.entropy_scale

        # ---- 总损失 ----
        loss = policy_loss + value_loss + entropy_loss

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪：防止梯度爆炸，最大范数限制为 0.5
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        # ---- 记录训练指标到 TensorBoard ----
        self.writer.add_scalar("loss_policy", policy_loss.item(), self.step_idx)
        self.writer.add_scalar("loss_value", value_loss.item(), self.step_idx)
        self.writer.add_scalar("loss_entropy", -entropy_loss.item(), self.step_idx)
        self.writer.add_scalar("loss_total", loss.item(), self.step_idx)
        self.writer.add_scalar("advantage", advantages.mean().item(), self.step_idx)
        self.writer.add_scalar("returns", returns.mean().item(), self.step_idx)

        self.step_idx += 1
        return loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item()

    def write_to_summary(self, name, value):
        """
        向 TensorBoard 写入自定义标量指标。

        Args:
            name: 指标名称，如 "eval_reward"
            value: 指标数值
        """
        self.writer.add_scalar(name, value, self.step_idx)

    def save(self):
        """
        保存模型检查点。

        保存内容包括：
        - step_idx: 当前训练步数（用于恢复训练进度）
        - policy_state_dict: 策略网络权重
        - optimizer_state_dict: 优化器状态（包含动量等信息）

        检查点保存路径格式：{run_dir}/ckpt/step{step_idx}.pt
        """
        path = os.path.join(self.ckpt_dir, f"step{self.step_idx}.pt")
        torch.save(
            {
                "step_idx": self.step_idx,
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"[INFO] Model checkpoint saved to {path}")
