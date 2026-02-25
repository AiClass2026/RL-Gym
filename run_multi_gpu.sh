#!/bin/bash
# ==============================================================================
# 多卡多进程训练启动脚本
# ==============================================================================
#
# 功能：
#   在多个 GPU 上启动独立的训练进程，每个进程运行在独立的 tmux 会话中。
#   这样可以：
#     - 充分利用多 GPU 的算力
#     - 每个进程独立训练（不同随机种子，增加多样性）
#     - 通过 tmux 随时查看/附加到任意训练进程
#     - 日志自动保存到文件，便于事后分析
#
# 用法：
#   chmod +x run_multi_gpu.sh
#   ./run_multi_gpu.sh
#
# 管理训练进程：
#   tmux list-sessions           # 查看所有训练会话
#   tmux attach -t train_gpu0_1  # 附加到 GPU0 的第 1 个进程
#   tmux kill-session -t train_gpu0_1  # 结束某个训练进程
# ==============================================================================

# ========== 配置区（根据你的环境修改）==========

GPU_IDS=(1)           # 要使用的 GPU ID 列表，多卡示例：(0 1 2 3)
PROCS_PER_GPU=4       # 每个 GPU 上启动的进程数（每个进程有独立的并行环境池）
NUM_ENVS=64           # 每个进程内的并行环境数量
EXTRA_ARGS=""         # 传给 train_ppo.py 的额外参数，如 "--sync --max_train_steps 100000"
CONDA_ENV="rl_gym"   # Conda 虚拟环境名称
CONDA_SH="/root/miniforge3/etc/profile.d/conda.sh"  # conda 初始化脚本路径

# PyTorch pip 包自带的 NVIDIA 库路径（优先于系统旧版 cuDNN）
# 这解决了系统 cuDNN 版本与 PyTorch 不匹配的问题
SITE_PKG="/root/miniforge3/envs/${CONDA_ENV}/lib/python3.11/site-packages"
NVIDIA_LIB="${SITE_PKG}/nvidia/cudnn/lib:${SITE_PKG}/nvidia/cublas/lib:${SITE_PKG}/nvidia/cuda_runtime/lib:${SITE_PKG}/nvidia/nvjitlink/lib"

# ========== 脚本逻辑（通常不需要修改）==========

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# 遍历每个 GPU 和每个进程编号，创建 tmux 会话
for gpu_id in "${GPU_IDS[@]}"; do
    for proc_idx in $(seq 1 "$PROCS_PER_GPU"); do
        # 会话命名格式：train_gpu{GPU编号}_{进程编号}
        session_name="train_gpu${gpu_id}_${proc_idx}"
        log_file="$LOG_DIR/${session_name}.log"

        # 如果同名会话已存在，先结束旧会话（避免重复）
        if tmux has-session -t "$session_name" 2>/dev/null; then
            echo "[KILL] 结束已有会话 $session_name"
            tmux kill-session -t "$session_name"
        fi

        # 启动新的 tmux 会话，在其中执行训练命令
        # CUDA_VISIBLE_DEVICES 控制该进程只看到指定的 GPU
        # --run_id 使用会话名，确保每个进程的输出目录不冲突
        # tee 同时输出到终端和日志文件
        echo "[START] 启动 $session_name (GPU $gpu_id, num_envs=$NUM_ENVS)"
        tmux new-session -d -s "$session_name" \
            "bash --norc -c 'source $CONDA_SH && conda activate $CONDA_ENV && cd $SCRIPT_DIR && export LD_LIBRARY_PATH=$NVIDIA_LIB:\$LD_LIBRARY_PATH && CUDA_VISIBLE_DEVICES=$gpu_id python train_ppo.py --num_envs $NUM_ENVS --run_id $session_name $EXTRA_ARGS 2>&1 | tee $log_file'; exec bash"
    done
done

# 打印启动摘要
echo ""
echo "已启动的 tmux 会话:"
tmux list-sessions 2>/dev/null | grep -E "train_gpu[0-9]+_[0-9]+" || true
echo ""
echo "日志目录: $LOG_DIR"
echo "附加到某个会话: tmux attach -t train_gpu0_1"
echo "查看所有会话:   tmux list-sessions"
