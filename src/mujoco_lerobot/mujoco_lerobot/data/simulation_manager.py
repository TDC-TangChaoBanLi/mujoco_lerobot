"""仿真时序编排器。

持有所有仿真子组件，对外提供 run_episode(controller) 接口。

时序（基于 dataset 配置的 use_recode_scale）：
  - 物理步：physics_dt（默认 1ms）
  - 子采样：sample_interval_s = 1 / (recode_hz × max_scale)
  - 策略步：policy_dt（默认 10ms）
  - 帧边界：frame_interval_s = 1 / recode_hz
  - 相机：按最快相机帧率渲染全部相机（camrender 并行）

性能要点：
  - 相机一次并行渲染全部相机
  - 状态采样写入预分配缓冲
  - 帧流式写入，不缓存整集
  - 仅 render 模式做真实时间节流与 viewer 同步
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import mujoco
import numpy as np

from ..configs.config_loader import SceneConfig
from ..configs.dataset_config import DatasetConfig
from ..simulate.actuators import build_actuator_mapping, apply_arm_action
from ..simulate.camera_renderer import CameraRenderer
from ..simulate.mujoco_wrapper import MujocoWrapper
from .controllers import Controller
from .observation_collector import ObservationCollector
from .recording import RecordingDecision
from .reset_manager import ResetManager

log = logging.getLogger(__name__)

VIEWER_FPS = 30  # viewer 同步帧率


class SimulationManager:
    def __init__(
        self,
        config: SceneConfig,
        dataset_cfg: DatasetConfig,
        *,
        render: bool = False,
        viewer_key_callback: Callable[[int], None] | None = None,
    ) -> None:
        self._config = config
        self._dataset_cfg = dataset_cfg
        self._render = render
        self._viewer_key_callback = viewer_key_callback

        # 冲突校验
        self._warnings = self._validate()

        self._mj = MujocoWrapper(config.task.scene_path, render=render)
        self._mj.open()
        if render:
            if viewer_key_callback is not None:
                self._mj.launch_viewer(key_callback=viewer_key_callback)
            else:
                self._mj.launch_viewer()
            v = config.view
            self._mj.set_viewer_camera(
                v.lookat, v.distance, v.elevation, v.azimuth,
            )

        self._renderer = CameraRenderer(config.cameras, self._mj.model, self._mj.data)
        self._reset_mgr = ResetManager(self._mj, config.robots, config.task.objects)
        self._collector = ObservationCollector(self._mj, dataset_cfg)

        # 时序参数
        self._pdt = config.sim.physics_dt
        self._adt = config.sim.policy_dt
        self._policy_steps = config.sim.policy_steps
        self._max_time = config.collection.max_time

        # 相机渲染调度：以最快相机帧率渲染全部相机
        self._cam_dt = min((1.0 / c.fps for c in config.cameras), default=float("inf"))
        self._cam_timer = 0.0

        # 预计算每臂 actuator id（arm / gripper）
        self._arm_act_ids, self._grip_act_ids = build_actuator_mapping(
            self._mj, config.robots
        )

    def _validate(self) -> list[str]:
        from ..configs.config_loader import validate_config_vs_model
        # 先加载模型做校验
        mj = MujocoWrapper(self._config.task.scene_path, render=False)
        mj.open()
        warns = validate_config_vs_model(self._config, mj.model)
        mj.close()
        return warns

    # ── 属性 ───────────────────────────────────────────

    @property
    def model(self) -> mujoco.MjModel:
        return self._mj.model

    @property
    def data(self) -> mujoco.MjData:
        return self._mj.data

    @property
    def warnings(self) -> list[str]:
        return self._warnings

    # ── 公开接口 ───────────────────────────────────────

    def run_episode(
        self,
        controller: Controller,
        *,
        max_time: float | None = None,
        frame_callback: Callable[[dict[str, Any], np.ndarray], None] | None = None,
    ) -> tuple[str, int]:
        """运行一条 episode，录制生命周期由 controller 的 recording_decision 控制。

        每策略步询问 ``controller.recording_decision(recording)``：
          START      开始录制（从此刻起采样 / 渲染相机 / 记录帧）
          SAVED      结束并保存本集
          DISCARDED  结束并丢弃本集
          QUIT       结束并退出采集（或 viewer 被关闭）
          None       继续

        返回 (result, frame_count)：
          result ∈ {"saved", "discarded", "quit"}
            - saved     — controller 判定保存（SAVED）
            - discarded — controller 判定丢弃（DISCARDED），或超时未明确结束
            - quit      — controller 判定退出（QUIT），或 viewer 被关闭
          frame_count  录制阶段实际写入的帧数（0 帧不产生有效保存）。

        未录制阶段（等待期，如遥操作就位）：不采样、不渲染相机，仅做策略级
        真实时间节流（~100Hz），保证人机交互反馈可感知。
        """
        t_max = max_time if max_time is not None else self._max_time
        dcfg = self._dataset_cfg
        frame_dt = dcfg.frame_interval_s
        sample_dt = dcfg.sample_interval_s

        # ── 重置 ──
        self._reset_mgr.reset(randomize_objects=True)
        controller.reset()
        self._collector.reset()
        self._cam_timer = 0.0

        recording = False
        t_sim = t_policy = t_frame = t_sample = t_viewer = 0.0
        n_policy = 0
        wall_start = time.perf_counter()
        frame_count = 0
        last_action = np.zeros(self._config.action_dim, dtype=np.float32)
        viewer_interval = 1.0 / VIEWER_FPS if self._render else float("inf")
        result: str = "discarded"  # 自然结束（超时）默认丢弃

        while t_sim < t_max:
            self._mj.step()
            t_sim += self._pdt
            t_policy += self._pdt
            t_frame += self._pdt
            t_sample += self._pdt
            t_viewer += self._pdt

            # ── 录制中：子采样 + 相机渲染 ──
            if recording:
                # ── 子采样 ──
                if t_sample >= sample_dt:
                    t_sample -= sample_dt
                    self._collector.sample()

                # ── 相机渲染（最快相机帧率，一次并行渲染全部）──
                self._cam_timer += self._pdt
                if self._cam_timer >= self._cam_dt:
                    self._cam_timer -= self._cam_dt
                    frames = self._renderer.render_all(self._mj.data)
                    self._collector.set_camera_frames(
                        {f.name: {"rgb": f.rgb, "depth": f.depth} for f in frames}
                    )

            # ── 策略步 ──
            if t_policy >= self._adt:
                t_policy -= self._adt
                n_policy += 1
                obs = {"arm_joint_pos": self._arm_joint_positions()}
                action = controller.step(obs)
                self._apply_action(action)
                if recording:
                    self._collector.update_last_action(action)
                last_action = action

                decision = controller.recording_decision(recording)
                if decision is RecordingDecision.START:
                    # 开始录制：清空可能的预采样，从此刻起计帧
                    recording = True
                    t_frame = t_sample = 0.0
                    self._collector.reset()
                    frames = self._renderer.render_all(self._mj.data)
                    self._collector.set_camera_frames(
                        {f.name: {"rgb": f.rgb, "depth": f.depth} for f in frames}
                    )
                elif decision is not None:
                    result = decision.value
                    break
                elif not recording:
                    # 等待阶段：策略级真实时间节流（~100Hz），
                    # 保证窗口/键盘交互反馈可感知
                    target = wall_start + n_policy * self._adt
                    now = time.perf_counter()
                    if target > now:
                        time.sleep(target - now)

            # ── 帧边界（recode，仅录制中）──
            if recording and t_frame >= frame_dt:
                t_frame -= frame_dt
                if self._collector.is_ready():
                    frame = self._collector.flush(self._config.task.task_id)
                    if frame_callback is not None:
                        frame_callback(frame, last_action)
                    frame_count += 1

            # ── viewer ──
            if self._render:
                if recording:
                    # 录制阶段真实时间节流（与 wall_start 对齐）
                    target = wall_start + t_sim
                    now = time.perf_counter()
                    if target > now:
                        time.sleep(target - now)
                if t_viewer >= viewer_interval:
                    t_viewer -= viewer_interval
                    self._mj.sync_viewer()
                    if not self._mj.is_viewer_running():
                        result = "quit"
                        break

        # 0 帧不算有效保存（不产生空集）
        if result != "quit" and frame_count <= 0:
            result = "discarded"
        return result, frame_count

    # ── 内部 ───────────────────────────────────────────

    def _arm_joint_positions(self) -> np.ndarray:
        """所有臂的关节位置（仅 arm joints，不含 gripper），供控制器 IK。"""
        parts = []
        for r in self._config.robots:
            parts.append(self._mj.joint_qpos(r.prefixed_arm_joints))
        return np.concatenate(parts).astype(np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        """将 flat action 写入 MuJoCo ctrl。

        action layout: [arm0_joints..., grip0..., arm1_joints..., grip1..., ...]
        """
        apply_arm_action(
            self._mj, self._config.robots,
            self._arm_act_ids, self._grip_act_ids, action,
        )

    def close(self) -> None:
        self._renderer.close()
        self._mj.close()
