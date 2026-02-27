## 目录结构

```
RL-Gym/
├── README.md              # 本文件，项目总览与使用说明
├── requirements.txt       # Python 依赖包列表
├── .gitignore             # Git 忽略规则
│
├── ppo/                   # PPO 算法核心模块
│   ├── __init__.py        #   包初始化，导出公共接口
│   ├── agent.py           #   ActorCritic 网络 + PPOAgent 训练代理
│   └── utils.py           #   工具函数：图像预处理、帧堆叠、GAE 计算
│
├── train_ppo.py           # 训练主脚本：解析参数、创建环境、执行训练循环
├── test.py                # 简单测试脚本：用固定动作运行环境并录制视频
├── run_multi_gpu.sh       # 多 GPU / 多进程训练启动脚本 (tmux)
│
├── runs/                  # 训练产物（自动生成，已 .gitignore）
│   └── <时间戳>_<运行ID>_<模型名>/
│       ├── ckpt/          #   模型检查点 (.pt 文件)
│       ├── logs/          #   TensorBoard 事件文件
│       └── videos/        #   评估时录制的视频
```


## 文件功能详解

### `ppo/agent.py` — 核心网络与训练逻辑

- **`ActorCritic`** 类：CNN 特征提取 + Actor（策略）头 + Critic（价值）头
  - 输入：堆叠帧 `(84, 84, 4)`
  - Actor 输出：连续动作的均值 + 标准差（高斯分布）
  - Critic 输出：状态价值 V(s)
- **`PPOAgent`** 类：封装训练、预测、保存/加载等操作
  - `predict()`: 根据状态推理动作
  - `train()`: 执行一步 PPO 参数更新
  - `save()`: 保存模型检查点

### `ppo/utils.py` — 工具函数

- **`preprocess_frame()`**: 将 96×96 RGB 图像裁剪为 84×84 灰度图，归一化到 [-1, 1]
- **`FrameStack`** 类：维护最近 N 帧的滑动窗口，提供堆叠状态
- **`compute_gae()`**: 计算 GAE 优势估计，用于 PPO 更新

### `train_ppo.py` — 训练主流程

1. 解析命令行参数（学习率、折扣因子、环境数等）
2. 创建并行环境（`AsyncVectorEnv`）
3. 循环采样 → 计算 GAE → PPO 更新
4. 定期评估并录制视频、保存检查点

### `run_multi_gpu.sh` — 多进程启动

通过 tmux 在多个 GPU 上并行启动训练进程，每个进程独立运行 `train_ppo.py`。

## 环境配置步骤

### 1) 创建并进入 Conda 环境

```bash
conda create -n rl-gym python=3.10 -y
conda activate rl-gym
python -V
```

### 2) 安装依赖

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### 3) 验证核心依赖是否可用

```bash
python -c "import torch, gymnasium; print('torch=', torch.__version__); print('gymnasium=', gymnasium.__version__)"
```

> 若使用 GPU，可额外检查：
>
> ```bash
> python -c "import torch; print('cuda_available=', torch.cuda.is_available())"
> ```

## 训练、评测运行流程

### 1) 启动训练

最小示例（自动选择设备）：

```bash
python train_ppo.py
```

常用参数示例（显式指定 GPU、并行环境数和训练步数）：

```bash
python train_ppo.py \
  --device cuda:0 \
  --num_envs 32 \
  --max_train_steps 100000 \
  --eval_interval 2000 \
  --save_interval 10000
```

训练输出目录说明：

- `runs/<时间戳>_<运行ID>_<模型名>/ckpt/`：模型检查点（`.pt`）
- `runs/<时间戳>_<运行ID>_<模型名>/logs/`：TensorBoard 日志
- `runs/<时间戳>_<运行ID>_<模型名>/videos/`：训练期间评估视频

### 2) 查看训练日志（可选）

```bash
tensorboard --logdir runs --port 6006
```

浏览器访问：`http://localhost:6006`

### 3) 运行离线评测

使用保存的 checkpoint 进行评测（采样策略）：

```bash
python eval_ppo.py \
  --ckpt runs/<你的run目录>/ckpt/<checkpoint>.pt \
  --mode sample \
  --num_episodes 10 \
```

评测输出目录：

- `eval_videos/<时间戳>_<checkpoint名>/`：每个 episode 的评测视频
