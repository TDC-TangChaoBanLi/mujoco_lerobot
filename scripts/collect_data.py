#!/usr/bin/env python3
"""一键数据采集 — auto（scripted teacher 自动）/ keyboard（键盘遥操作）。

用法:
    # 列出任务
    uv run python scripts/collect_data.py --task-config configs/tasks/tasks.yaml --task list

    # 自动采集
    uv run python scripts/collect_data.py \
        --task-config configs/tasks/tasks.yaml \
        --task pick_place \
        --dataset-config configs/dataset/dataset_pick_place.yaml \
        --episodes 50

    # 键盘采集（mujoco viewer 窗口）
    uv run python scripts/collect_data.py \
        --task-config configs/tasks/tasks.yaml \
        --task dual_pick_place \
        --dataset-config configs/dataset/dataset_dual_pick_place.yaml \
        --mode keyboard

    # PushT 鼠标遥操作采集（push_t 任务在 teacher 模式下自动启动
    # pygame 2D 俯视窗口，无需单独模式；需图形界面 DISPLAY）
    uv run python scripts/collect_data.py \
        --task-config configs/tasks/tasks.yaml \
        --task push_t \
        --dataset-config configs/dataset/dataset_push_t.yaml \
        --episodes 20

    # 补充数据：在已有数据集上追加（先校验配置一致，不一致会报错）。
    # --append 下 --episodes 表示"本轮新增条数"：总量 = 已有 + 新增
    uv run python scripts/collect_data.py \
        --task-config configs/tasks/tasks.yaml \
        --task pick_place \
        --dataset-config configs/dataset/dataset_pick_place.yaml \
        --append outputs/datasets/pick_place/20260808_184250 \
        --episodes 20 \
        --no-render

    # 自定义 repo_id 与数据集输出目录（--output-dir 为精确目录，
    # 不再自动附加 <任务名>/<时间戳> 子目录）
    uv run python scripts/collect_data.py \
        --task-config configs/tasks/tasks.yaml \
        --task pick_place \
        --dataset-config configs/dataset/dataset_pick_place.yaml \
        --repo-id my_org/my_pick_place \
        --output-dir outputs/datasets/my_pick_place \
        --episodes 50 \
        --no-render

键盘控制（keyboard 模式）：
    W/S A/D R/F    x / y / z 平移
    方向键 Q/E     俯仰/偏航
    Space          夹爪开合
    Enter          保存当前 episode
    Backspace      丢弃当前 episode
    Esc            退出
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("FFMPEG_LOGLEVEL", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# 无头环境（无 DISPLAY，如 SSH 服务器）：使用 EGL 离屏渲染（需 GPU + EGL 驱动），
# 避免 GLFW 因缺少显示而报错；有 DISPLAY 时保持默认（viewer 可正常使用）。
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
for _name in ("lerobot", "datasets", "PIL", "torchvision", "ffmpeg", "av", "x265"):
    logging.getLogger(_name).setLevel(logging.WARNING)

from mujoco_lerobot.configs import (
    load_scene_config,
    get_task_list,
    load_teacher_config,
    resolve_config_path,
)
from mujoco_lerobot.configs.dataset_config import DatasetConfig
from mujoco_lerobot.data import (
    SimulationManager,
    ScriptedTeacherController,
    KeyboardTeleopController,
    LeRobotDatasetWriter,
    LeRobotDatasetConfig,
)

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MuJoCo 数据采集（auto / keyboard）")
    p.add_argument("--task-config", default="configs/tasks/tasks.yaml",
                   help="任务配置文件路径")
    p.add_argument("--task", required=True, help="任务名（--task list 列出所有）")
    p.add_argument("--dataset-config", default=None,
                   help="数据集记录格式配置文件路径（--task list 时不需要）")
    p.add_argument("--mode", choices=["teacher", "keyboard"],
                   default="teacher",
                   help="采集模式（keyboard 为 mujoco viewer 键盘遥操作；"
                        "push_t 任务在 teacher 模式下自动启动鼠标遥操作窗口）")
    p.add_argument("--episodes", type=int, default=50,
                   help="目标 episode 数（--append 时为本轮新增条数，总量 = 已有 + 新增）")
    p.add_argument("--max-time", type=float, default=None,
                   help="每 episode 仿真时间上限（秒），默认取 simulate_default 配置")
    p.add_argument("--repo-id", default=None,
                   help="数据集 repo_id（默认 mujoco_<任务名>，如 mujoco_pick_place）")
    p.add_argument("--output", default="outputs/datasets", help="输出根目录")
    p.add_argument("--output-dir", default=None,
                   help="直接指定数据集输出目录（精确路径，不再自动附加 <任务名>/<时间戳> 子目录；"
                        "默认在 --output 下按 <任务名>/<时间戳> 组织）")
    p.add_argument("--append", default=None, metavar="DIR",
                   help="补充数据：在已有数据集目录上追加 episodes"
                        "（追加前校验当前配置与原有记录配置一致）")
    p.add_argument("--no-render", action="store_true", help="无头模式（不打开 viewer）")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出目录")
    return p.parse_args()


def build_writer(args, scene_cfg, dataset_cfg) -> LeRobotDatasetWriter:
    wcfg = LeRobotDatasetConfig(
        repo_id=args.repo_id or f"mujoco_{scene_cfg.task.name}",
        root=None,  # 下方按模式填充
        fps=int(round(dataset_cfg.recode_hz)),
        cameras=scene_cfg.cameras,
    )

    if args.append:
        # 补充数据：在已有数据集上追加（先做配置一致性检查，不一致会抛错）
        wcfg.root = Path(args.append).expanduser().resolve()
        return LeRobotDatasetWriter.resume(wcfg, dataset_cfg=dataset_cfg)

    # 新建数据集：默认按 <output>/<task>/<时间戳> 组织；
    # 指定 --output-dir 时直接写入该精确目录（不再附加任务名/时间戳子目录）
    if args.output_dir:
        root = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(args.output) / scene_cfg.task.name / ts
    wcfg.root = root.resolve()
    return LeRobotDatasetWriter.create_new(
        wcfg, dataset_cfg=dataset_cfg, overwrite=args.overwrite
    )


def collect_teacher(
    args,
    scene_cfg,
    dataset_cfg,
    writer,
) -> None:
    """教师采集：teacher 反复采集，成功保存，否则丢弃（可重试）。

    完全任务无关：
    - 录制生命周期由 teacher 的 ``recording_decision`` 控制（run_episode 内）
    - 采集会话钩子 ``start_collection`` / ``end_collection`` 由 teacher 实现
      （遥操作类 teacher 在此启动/关闭交互窗口，如 PushT 的 pygame 2D 窗口）
    - 每 episode 尝试上限取 ``controller.retry_limit``（None = 场景配置
      ``collection.max_attempts``；遥操作 teacher 通常为 1，用户丢弃即下一集）
    """
    teacher_cfg = load_teacher_config(scene_cfg.task.teacher_config_file)
    mgr = SimulationManager(scene_cfg, dataset_cfg, render=not args.no_render)
    ctrl = ScriptedTeacherController(scene_cfg, teacher_cfg, mgr.model, mgr.data)

    max_attempts = scene_cfg.collection.max_attempts
    t0 = time.perf_counter()
    collected = 0
    total_attempts = 0

    try:
        ctrl.start_collection()  # teacher 钩子：如 PushT 启动遥操作窗口
        while collected < args.episodes:
            cb, flush, discard = writer.make_stream_callback(
                task_label=scene_cfg.task.name
            )

            success = False
            attempts = ctrl.retry_limit or max_attempts
            for _attempt in range(1, attempts + 1):
                try:
                    t_ep = time.perf_counter()
                    result, n_frames = mgr.run_episode(
                        ctrl,
                        max_time=args.max_time,
                        frame_callback=cb,
                    )
                except KeyboardInterrupt:
                    # 中断（如 Ctrl+C）：丢弃当前未完成 episode 的所有帧，
                    # 避免把半截/失败数据写进数据集。
                    discard()
                    print("  ✗ 中断，已丢弃当前 episode 未保存的帧")
                    raise
                total_attempts += 1
                ep_wall = time.perf_counter() - t_ep

                if result == "quit":
                    print("  已退出采集")
                    return
                if result == "saved":
                    flush()
                    collected += 1
                    success = True
                    print(
                        f"  ✓ {collected}/{args.episodes}  frames={n_frames}  "
                        f"ep_wall={ep_wall:.1f}s  "
                        f"total={time.perf_counter()-t0:.0f}s"
                    )
                    break
                else:
                    # 失败（或用户丢弃）：本集帧不入库
                    discard()
                    print(
                        f"  ✗ 尝试 {total_attempts} 失败  frames={n_frames}  "
                        f"ep_wall={ep_wall:.1f}s  (帧已丢弃，不入库)"
                    )

            if not success:
                print(
                    f"  ⚠ 连续 {attempts} 次失败，跳过本集，继续下一集"
                )
    finally:
        ctrl.end_collection()  # teacher 钩子：如 PushT 关闭遥操作窗口
        mgr.close()

    elapsed = time.perf_counter() - t0
    epm = collected / elapsed * 60 if elapsed > 0 else 0
    print(f"完成: {collected} episodes 于 {elapsed:.1f}s "
          f"({epm:.1f} ep/min, 总尝试 {total_attempts})")


def collect_keyboard(args, scene_cfg, dataset_cfg, writer) -> None:
    """键盘遥操作采集：mujoco viewer 实时控制。"""
    holder: dict = {}  # 延迟绑定 ctrl（mgr 构造时 ctrl 尚未创建）

    def key_cb(key: int) -> None:
        ctrl = holder.get("ctrl")
        if ctrl is not None:
            # 录制控制键（Enter/Backspace/Esc）走 record_event，其余走移动键
            if not ctrl.record_event(key):
                ctrl.key_event(key)

    mgr = SimulationManager(
        scene_cfg, dataset_cfg, render=True, viewer_key_callback=key_cb
    )
    ctrl = KeyboardTeleopController(scene_cfg, mgr.model, mgr.data)
    holder["ctrl"] = ctrl

    print("\n" + "=" * 60)
    print(f"键盘遥操作采集  任务: {scene_cfg.task.name}")
    print("  W/S A/D R/F = x/y/z  方向键/Q/E = 旋转  Space = 夹爪")
    print("  Enter=保存  Backspace=丢弃  Esc=退出")
    print("=" * 60)

    collected = 0
    t0 = time.perf_counter()
    try:
        while collected < args.episodes:
            cb, flush, discard = writer.make_stream_callback(
                task_label=scene_cfg.task.name
            )
            result, n_frames = mgr.run_episode(
                ctrl,
                max_time=args.max_time,
                frame_callback=cb,
            )
            if result == "quit":
                print("  已退出采集")
                break
            if result == "saved":
                flush()
                collected += 1
                print(f"  ✓ 保存 episode {collected}/{args.episodes}  "
                      f"frames={n_frames}  total={time.perf_counter()-t0:.0f}s")
            else:
                discard()
                print(f"  ✗ 丢弃本集 (frames={n_frames})")
    finally:
        mgr.close()

    elapsed = time.perf_counter() - t0
    print(f"完成: {collected} episodes 于 {elapsed:.1f}s")


def main() -> None:
    args = parse_args()

    if args.task == "list":
        print("可用任务:", get_task_list(args.task_config))
        return

    if not args.dataset_config:
        raise SystemExit("--dataset-config 必填（--task list 除外）")

    scene_cfg = load_scene_config(args.task, args.task_config)
    dataset_cfg = DatasetConfig.from_yaml(args.dataset_config)

    # 打印配置摘要
    print("=" * 72)
    print(f"任务: {args.task} | 模式: {args.mode}")
    print(f"  场景: {scene_cfg.task.scene_file}")
    print(f"  记录: {dataset_cfg.recode_hz:.0f} Hz  "
          f"use_recode_scale={dataset_cfg.use_recode_scale}  "
          f"max_scale={dataset_cfg.max_scale}")
    print(f"  repo_id: {args.repo_id or f'mujoco_{args.task}'}")
    print(f"  臂: {[r.name for r in scene_cfg.robots]}  "
          f"相机: {[c.name for c in scene_cfg.cameras]}")
    print(f"  state_dim={dataset_cfg.state_dim}  "
          f"action_dim={dataset_cfg.action_dim}")
    print("=" * 72)

    for w in SimulationManager(scene_cfg, dataset_cfg, render=False).warnings:
        print(f"  [warn] {w}")

    writer = build_writer(args, scene_cfg, dataset_cfg)
    if args.append:
        n_existing = writer.dataset.meta.total_episodes
        print(f"  追加模式: 已有 {n_existing} episodes, "
              f"本次新增 {args.episodes} → 目标 {n_existing + args.episodes}")
    else:
        print(f"  新建数据集 → {writer.config.resolved_root()}")
    try:
        if args.mode == "teacher":
            collect_teacher(args, scene_cfg, dataset_cfg, writer)
        else:
            collect_keyboard(args, scene_cfg, dataset_cfg, writer)
    finally:
        writer.finalize()
        print(f"  数据集 → {writer.config.resolved_root()}")


if __name__ == "__main__":
    main()
