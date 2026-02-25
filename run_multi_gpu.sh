#!/bin/bash
# 多卡多进程训练脚本：每个进程在独立 tmux 会话中运行
# 用法: ./run_multi_gpu.sh

# ========== 配置区 ==========
GPU_IDS=(0)         # 要使用的 GPU ID 列表
PROCS_PER_GPU=4       # 每个 GPU 上启动的进程数
NUM_ENVS=64           # 每个进程的并行环境数
EXTRA_ARGS=""         # 额外参数，如 "--sync --max_train_steps 100000"
CONDA_ENV="rl_gym"
CONDA_SH="/root/miniforge3/etc/profile.d/conda.sh"

# PyTorch pip 包自带的 NVIDIA 库路径（优先于系统旧版 cuDNN）
SITE_PKG="/root/miniforge3/envs/${CONDA_ENV}/lib/python3.11/site-packages"
NVIDIA_LIB="${SITE_PKG}/nvidia/cudnn/lib:${SITE_PKG}/nvidia/cublas/lib:${SITE_PKG}/nvidia/cuda_runtime/lib:${SITE_PKG}/nvidia/nvjitlink/lib"

# ========== 脚本逻辑 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

for gpu_id in "${GPU_IDS[@]}"; do
    for proc_idx in $(seq 1 "$PROCS_PER_GPU"); do
        session_name="train_gpu${gpu_id}_${proc_idx}"
        log_file="$LOG_DIR/${session_name}.log"
        if tmux has-session -t "$session_name" 2>/dev/null; then
            echo "[KILL] 结束已有会话 $session_name"
            tmux kill-session -t "$session_name"
        fi
        echo "[START] 启动 $session_name (GPU $gpu_id, num_envs=$NUM_ENVS)"
        tmux new-session -d -s "$session_name" \
            "bash --norc -c 'source $CONDA_SH && conda activate $CONDA_ENV && cd $SCRIPT_DIR && export LD_LIBRARY_PATH=$NVIDIA_LIB:\$LD_LIBRARY_PATH && CUDA_VISIBLE_DEVICES=$gpu_id python train_ppo.py --num_envs $NUM_ENVS --run_id $session_name $EXTRA_ARGS 2>&1 | tee $log_file'; exec bash"
    done
done

echo ""
echo "已启动的 tmux 会话:"
tmux list-sessions 2>/dev/null | grep -E "train_gpu[0-9]+_[0-9]+" || true
echo ""
echo "日志目录: $LOG_DIR"
echo "附加到某个会话: tmux attach -t train_gpu0_1"
echo "查看所有会话:   tmux list-sessions"
