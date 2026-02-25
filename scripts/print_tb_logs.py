#!/usr/bin/env python3
"""
TensorBoard 日志文本打印工具
==============================
在命令行中以结构化表格形式打印 runs/ 目录下所有实验的 TensorBoard 标量数据。

适用场景：
    - 服务器没有图形界面、无法打开浏览器使用 TensorBoard
    - 快速对比多个实验的训练指标
    - 数据量过大时自动均匀采样，保留首尾数据点

实现方式：
    直接读取 TFRecord 二进制文件并用 protobuf 解析，
    跳过 EventAccumulator 的开销，速度更快。
    使用多进程并行解析多个实验目录。

用法：
    python print_tb_logs.py [ROOT_DIR]
    python print_tb_logs.py /path/to/runs --max-points 100
"""

import os
import sys
import glob
import struct
import time
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tensorboard.compat.proto.event_pb2 import Event

# 默认值
DEFAULT_RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
DEFAULT_MAX_POINTS = 50
MAX_POINTS = DEFAULT_MAX_POINTS


def read_tfrecord(path):
    """
    逐条读取 TFRecord 文件中的原始记录。

    TFRecord 是 TensorFlow/TensorBoard 使用的二进制存储格式，每条记录的结构为：
        [8 字节 length] [4 字节 CRC] [data] [4 字节 CRC]

    Args:
        path: TFRecord 文件路径

    Yields:
        bytes: 每条记录的原始二进制数据（Event 的 protobuf 序列化结果）
    """
    with open(path, "rb") as f:
        while True:
            # 读取头部：8 字节无符号 64 位整数（数据长度）+ 4 字节 CRC 校验
            header = f.read(12)
            if len(header) < 12:
                break
            # '<Q' 表示小端序的 unsigned long long
            data_len = struct.unpack("<Q", header[:8])[0]
            data = f.read(data_len)
            if len(data) < data_len:
                break
            f.read(4)  # 跳过数据 CRC
            yield data


def parse_experiment(exp_dir):
    """
    解析单个实验目录中的所有 TensorBoard 事件文件。

    Args:
        exp_dir: 实验目录路径，如 runs/20260224_184840_CarRacing-v3/

    Returns:
        (exp_name, tag_data) 元组：
        - exp_name: 实验目录名
        - tag_data: 字典 {标签名: [(step, value), ...]}
    """
    exp_name = os.path.basename(exp_dir)

    # 递归查找所有 TensorBoard 事件文件
    event_files = sorted(glob.glob(os.path.join(exp_dir, "**", "events.out.tfevents.*"), recursive=True))
    if not event_files:
        return exp_name, {}

    # 解析每个事件文件，提取标量数据
    tag_data = defaultdict(list)
    for ef in event_files:
        for raw in read_tfrecord(ef):
            event = Event()
            event.ParseFromString(raw)
            # 只关注包含 summary 的事件（标量指标）
            if event.HasField("summary"):
                for v in event.summary.value:
                    if v.HasField("simple_value"):
                        tag_data[v.tag].append((event.step, v.simple_value))

    return exp_name, dict(tag_data)


def sample(points, max_n):
    """
    对数据点列表进行均匀采样，保留首尾。

    当数据点过多时，等间距选取 max_n 个点，确保首个和末尾数据点始终保留。

    Args:
        points: 数据点列表
        max_n: 最大保留数量

    Returns:
        采样后的数据点列表
    """
    if len(points) <= max_n:
        return points
    step = (len(points) - 1) / (max_n - 1)
    indices = sorted(set(int(round(i * step)) for i in range(max_n)))
    return [points[i] for i in indices]


def fmt(v):
    """
    格式化数值为字符串，自动选择定点或科学计数法。

    Args:
        v: 数值

    Returns:
        格式化后的字符串
    """
    if abs(v) >= 1e4 or (0 < abs(v) < 1e-3):
        return f"{v:.4e}"
    return f"{v:.4f}"


def print_experiment(exp_name, tag_data):
    """
    以表格形式打印单个实验的所有标量指标。

    Args:
        exp_name: 实验名称
        tag_data: 标签数据字典 {tag: [(step, value), ...]}
    """
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  实验: {exp_name}")
    print(sep)

    for tag in sorted(tag_data):
        points = tag_data[tag]
        total = len(points)
        sampled = sample(points, MAX_POINTS)
        flag = f"，采样 {len(sampled)} 条" if total > MAX_POINTS else ""

        print(f"\n  ── {tag}  (共 {total} 条{flag})")
        print(f"  {'Step':>10s}  {'Value':>14s}")
        print(f"  {'─'*10}  {'─'*14}")
        for step, val in sampled:
            print(f"  {step:>10d}  {fmt(val):>14s}")


def find_experiment_dirs(root_dir):
    """
    递归扫描根目录，找出所有包含 TensorBoard 事件文件的目录。

    从 root_dir 开始向下搜索，将包含事件文件的最浅层目录视为一个实验。

    Args:
        root_dir: 扫描的根目录

    Returns:
        排序后的实验目录列表
    """
    event_files = glob.glob(os.path.join(root_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        return []

    # 收集所有事件文件所在的目录，然后找到相对于 root_dir 的最顶层实验目录
    exp_dirs = set()
    for ef in event_files:
        rel = os.path.relpath(os.path.dirname(ef), root_dir)
        top_level = rel.split(os.sep)[0] if rel != "." else "."
        if top_level == ".":
            exp_dirs.add(root_dir)
        else:
            exp_dirs.add(os.path.join(root_dir, top_level))

    return sorted(exp_dirs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="在命令行中打印 TensorBoard 标量日志数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=DEFAULT_RUNS_DIR,
        help=f"日志文件的根目录，递归扫描其下的事件文件（默认: runs/）",
    )
    parser.add_argument(
        "--max-points", "-n",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help=f"每个标量标签最多显示的数据点数（默认: {DEFAULT_MAX_POINTS}）",
    )
    return parser.parse_args()


def main():
    """
    主函数：扫描指定根目录下所有实验，并行解析后打印结果。
    """
    args = parse_args()
    root_dir = os.path.abspath(args.root_dir)
    global MAX_POINTS
    MAX_POINTS = args.max_points

    if not os.path.isdir(root_dir):
        print(f"错误: 目录不存在: {root_dir}", file=sys.stderr)
        sys.exit(1)

    # 递归扫描实验目录
    exp_dirs = find_experiment_dirs(root_dir)
    if not exp_dirs:
        print(f"在 {root_dir} 下未找到任何 TensorBoard 事件文件。")
        return

    print(f"扫描目录: {root_dir}")
    print(f"共 {len(exp_dirs)} 个实验，每字段最多 {MAX_POINTS} 个采样点。正在并行解析...",
          flush=True)

    # 使用多进程并行解析（最多 4 个工作进程）
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=min(4, len(exp_dirs))) as pool:
        futures = {pool.submit(parse_experiment, d): d for d in exp_dirs}
        for fut in as_completed(futures):
            name, data = fut.result()
            results[name] = data
            print(f"  ✓ {name} ({sum(len(v) for v in data.values())} 条记录)", flush=True)

    elapsed = time.time() - t0
    print(f"\n解析完成，耗时 {elapsed:.1f}s\n")

    # 按目录顺序打印结果
    for exp_dir in exp_dirs:
        name = os.path.basename(exp_dir)
        if name in results and results[name]:
            print_experiment(name, results[name])

    print()


if __name__ == "__main__":
    main()
