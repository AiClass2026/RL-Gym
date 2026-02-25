"""PPO Agent: CNN + Actor-Critic 网络"""
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


class ActorCritic(nn.Module):
    """共享 CNN + Actor + Critic 分支"""

    def __init__(self, input_shape, num_actions, action_min, action_max):
        super().__init__()
        self.num_actions = num_actions
        self.action_min = torch.tensor(action_min, dtype=torch.float32)
        self.action_max = torch.tensor(action_max, dtype=torch.float32)
        self.register_buffer("_action_min", self.action_min)
        self.register_buffer("_action_max", self.action_max)

        # CNN: input (N, 4, 84, 84)
        self.conv1 = nn.Conv2d(input_shape[-1], 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        # 计算 flatten 后的维度: 84->20->9, 4->20->9
        def conv_out_size(size, kernel, stride):
            return (size - kernel) // stride + 1

        h = conv_out_size(conv_out_size(input_shape[0], 8, 4), 3, 2)
        w = conv_out_size(conv_out_size(input_shape[1], 8, 4), 3, 2)
        self.feature_dim = 32 * h * w

        self.fc = nn.Linear(self.feature_dim, 256)

        # Actor: mean + log_std
        self.actor_mean = nn.Linear(256, num_actions)
        self.actor_logstd = nn.Parameter(torch.full((num_actions,), np.log(0.4)))

        # Critic
        self.critic = nn.Linear(256, 1)

    def _forward_features(self, x):
        """x: (N, H, W, C) -> (N, C, H, W) for conv"""
        if x.shape[-1] == 4:  # channels last
            x = x.permute(0, 3, 1, 2).contiguous()
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = F.leaky_relu(self.fc(x))
        return x

    def forward(self, x):
        features = self._forward_features(x)
        mean_raw = torch.tanh(self.actor_mean(features))
        action_min = self._action_min.to(x.device)
        action_max = self._action_max.to(x.device)
        mean = action_min + (mean_raw + 1) / 2 * (action_max - action_min)
        std = torch.exp(self.actor_logstd).expand_as(mean)
        value = self.critic(features).squeeze(-1)
        return mean, std, value

    def get_action_and_value(self, x, action=None):
        mean, std, value = self.forward(x)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value


class PPOAgent:
    """PPO 训练代理"""

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
        self.epsilon = epsilon
        self.value_scale = value_scale
        self.entropy_scale = entropy_scale
        self.step_idx = 0

        self.run_dir = run_dir
        self.ckpt_dir = os.path.join(run_dir, "ckpt")
        self.log_dir = os.path.join(run_dir, "logs")
        self.video_dir = os.path.join(run_dir, "videos")

        for d in [self.ckpt_dir, self.log_dir, self.video_dir]:
            os.makedirs(d, exist_ok=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.policy = ActorCritic(input_shape, num_actions, action_min, action_max).to(self.device)
        self.policy_old = ActorCritic(input_shape, num_actions, action_min, action_max).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.writer = SummaryWriter(self.log_dir)

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        return x

    def predict(self, states, use_old_policy=False, greedy=False):
        policy = self.policy_old if use_old_policy else self.policy
        x = self._to_tensor(np.array(states))
        with torch.no_grad():
            mean, std, value = policy(x)
            if greedy:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
            action = torch.clamp(action, policy._action_min, policy._action_max)
        action_np = action.cpu().numpy()
        value_np = value.cpu().numpy()
        return action_np, value_np

    def update_old_policy(self):
        self.policy_old.load_state_dict(self.policy.state_dict())

    def train(self, states, taken_actions, returns, advantages, learning_rate):
        states = self._to_tensor(states)
        taken_actions = self._to_tensor(taken_actions)
        returns = self._to_tensor(returns)
        advantages = self._to_tensor(advantages)

        for p in self.optimizer.param_groups:
            p["lr"] = learning_rate(self.step_idx) if callable(learning_rate) else learning_rate

        _, log_prob_new, entropy, value = self.policy.get_action_and_value(states, taken_actions)
        with torch.no_grad():
            _, log_prob_old, _, _ = self.policy_old.get_action_and_value(states, taken_actions)

        ratio = torch.exp(log_prob_new - log_prob_old)
        policy_loss_1 = ratio * advantages
        policy_loss_2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        value_loss = F.mse_loss(value, returns) * self.value_scale
        entropy_loss = -entropy.mean() * self.entropy_scale
        loss = policy_loss + value_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        self.writer.add_scalar("loss_policy", policy_loss.item(), self.step_idx)
        self.writer.add_scalar("loss_value", value_loss.item(), self.step_idx)
        self.writer.add_scalar("loss_entropy", -entropy_loss.item(), self.step_idx)
        self.writer.add_scalar("loss_total", loss.item(), self.step_idx)
        self.writer.add_scalar("advantage", advantages.mean().item(), self.step_idx)
        self.writer.add_scalar("returns", returns.mean().item(), self.step_idx)

        self.step_idx += 1
        return loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item()

    def write_to_summary(self, name, value):
        self.writer.add_scalar(name, value, self.step_idx)

    def save(self):
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
