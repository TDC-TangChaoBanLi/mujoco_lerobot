"""MuJoCo Gymnasium 环境 — LeRobot 兼容。

观测键使用 gym 风格（无 `observation.` 前缀）：
  - state.<source>   （多速率，shape (R, D)）
  - images.<cam>.rgb / .depth

lerobot-eval 的通用 preprocess_observation 会把 `observation.` 前缀补上，
随后经 EnvConfig.get_env_processors 的处理器（默认恒等）适配策略。
任何在该数据集格式上训练的策略可直接评估；其他策略可配合外部观测数据处理器适配。

时序结构：
  - 每个 `step()` = 一个记录帧（recode 间隔）
  - 帧内按 dataset 的 use_recode_scale 对状态进行多速率子采样 → obs shape (R, D)
  - 相机每帧渲染一次（camrender 并行 + 深度线性化）

本环境是任务无关的：任务名 + dataset 配置全部由构造参数驱动，
成功判定经由 teacher 注册表按任务自动发现（见 ``_init_teacher``）。
"""

from __future__ import annotations

import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mujoco_lerobot.configs.config_loader import load_scene_config
from mujoco_lerobot.configs.dataset_config import DatasetConfig
from mujoco_lerobot.configs.teacher_config import load_teacher_config
from mujoco_lerobot.simulate.actuators import build_actuator_mapping, apply_arm_action
from mujoco_lerobot.simulate.camera_renderer import CameraRenderer
from mujoco_lerobot.simulate.mujoco_wrapper import MujocoWrapper
from mujoco_lerobot.data.observation_collector import ObservationCollector
from mujoco_lerobot.data.reset_manager import ResetManager
from mujoco_lerobot.data.teachers import create_teacher, TEACHER_REGISTRY

log = logging.getLogger(__name__)


class MultiTeacherWrapper:
    """聚合多个单臂 teacher：成功判定 = 所有 teacher 均判定成功。"""

    def __init__(self) -> None:
        self._teachers: list = []

    def add(self, teacher) -> None:
        self._teachers.append(teacher)

    def check_success(self) -> bool:
        if not self._teachers:
            return False
        return all(t.check_success() for t in self._teachers)


class MujocoLerobotEnv(gym.Env):
    """MuJoCo 多臂多相机环境 — LeRobot 兼容（dataset 配置驱动）。"""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        task_name: str = "pick_place",
        dataset_config: str = "configs/dataset/dataset_pick_place.yaml",
        render_mode: str | None = None,
        max_episode_steps: int | None = None,
    ) -> None:
        super().__init__()
        self._task_name = task_name
        self._dataset_config_path = str(dataset_config)
        self._render_mode = render_mode
        self._max_episode_steps = (
            max_episode_steps if max_episode_steps is not None else 300
        )

        # 配置（惰性加载模型）
        self._scene_cfg = load_scene_config(task_name)
        self._dataset_cfg = DatasetConfig.from_yaml(dataset_config)

        # 惰性初始化
        self._mj: MujocoWrapper | None = None
        self._renderer: CameraRenderer | None = None
        self._reset_mgr: ResetManager | None = None
        self._arm_act_ids: dict = {}
        self._grip_act_ids: dict = {}
        self._readers: dict[str, callable] = {}
        self._cam_ids: dict[str, int] = {}
        self._step_count = 0

        # 评估成功判定：由任务对应 teacher 的 check_success() 判定（自适应）
        self._teacher = None

        # 视频帧率：一个仿真步 = 一个记录帧，视频按 recode_hz 播放则时长=仿真时长
        self.metadata = dict(self.metadata)
        self.metadata["render_fps"] = int(round(self._dataset_cfg.recode_hz))

        self._build_spaces()

    # ── 空间构建 ───────────────────────────────────────

    def _build_spaces(self) -> None:
        dc = self._dataset_cfg
        obs_dict: dict[str, spaces.Space] = {}

        if dc.flat_mode:
            # 扁平模式：state 拼接为单个 (state_dim,) 向量（ACT 兼容）
            obs_dict["state"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(dc.state_dim,), dtype=np.float32,
            )
        else:
            for src in dc.state_sources():
                obs_dict[src.name] = spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(src.num_subs, src.dim_per_sub),
                    dtype=np.float32,
                )

        for cam in self._scene_cfg.cameras:
            obs_dict[f"images.{cam.name}.rgb"] = spaces.Box(
                low=0, high=255, shape=(3, cam.height, cam.width), dtype=np.uint8,
            )
            obs_dict[f"images.{cam.name}.depth"] = spaces.Box(
                low=0.0, high=np.inf, shape=(1, cam.height, cam.width),
                dtype=np.float32,
            )

        self.observation_space = spaces.Dict(obs_dict)

        # 动作：flat 关节目标（臂 + 夹爪）
        self.action_dim = self._scene_cfg.action_dim
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32,
        )

    # ── 生命周期 ───────────────────────────────────────

    def _ensure_env(self) -> None:
        """延迟初始化 MuJoCo + 渲染器（兼容 AsyncVectorEnv 的 fork 机制）。"""
        if self._mj is not None:
            return

        use_viewer = self._render_mode == "human"
        self._mj = MujocoWrapper(self._scene_cfg.task.scene_path, render=use_viewer)
        self._mj.open()
        if use_viewer:
            v = self._scene_cfg.view
            self._mj.set_viewer_camera(
                v.lookat, v.distance, v.elevation, v.azimuth,
            )

        self._renderer = CameraRenderer(
            self._scene_cfg.cameras, self._mj.model, self._mj.data
        )
        self._arm_act_ids, self._grip_act_ids = build_actuator_mapping(
            self._mj, self._scene_cfg.robots
        )

        # 复位管理器（与数据采集一致：机械臂到位 + 物体随机化）
        self._reset_mgr = ResetManager(
            self._mj, self._scene_cfg.robots, self._scene_cfg.task.objects
        )

        # 评估成功判定：实例化任务对应 teacher，调用其 check_success()（自适应）
        self._init_teacher()

        # 状态读取器（复用采集器的工厂函数）
        mj = self._mj
        self._readers = {}
        self._site_prev: dict[str, np.ndarray] = {}
        for src in self._dataset_cfg.state_sources():
            if src.source_type == "joint_pos":
                adrs = [mj._jnt_qposadr.get(n, -1) for n in src.read_names]
                self._readers[src.name] = ObservationCollector._make_qpos_reader(
                    adrs, src.dim_per_sub
                )
            elif src.source_type == "joint_vel":
                adrs = [mj._jnt_dofadr.get(n, -1) for n in src.read_names]
                self._readers[src.name] = ObservationCollector._make_qvel_reader(
                    adrs, src.dim_per_sub
                )
            elif src.source_type == "joint_effort":
                adrs = [mj._jnt_dofadr.get(n, -1) for n in src.read_names]
                self._readers[src.name] = ObservationCollector._make_effort_reader(
                    adrs, src.dim_per_sub
                )
            elif src.source_type.startswith("sensor."):
                slices = [mj.sensor_slice(n) for n in src.read_names]
                self._readers[src.name] = ObservationCollector._make_sensor_reader(
                    slices, src.dim_per_sub
                )
            elif src.source_type.startswith("site_"):
                suffix = src.source_type.rsplit("_", 1)[-1]
                sids = [mj._site_ids.get(n, -1) for n in src.read_names]
                fid = mj._site_ids.get(src.frame_site) if src.frame_site != "world" else None
                per = 3 if suffix != "quat" else 4
                prev = np.zeros(len(sids) * per, dtype=np.float32)
                self._site_prev[src.name] = prev
                self._readers[src.name] = ObservationCollector._make_site_reader(
                    sids, suffix, src.dim_per_sub, fid,
                    src.pose_type, self._dataset_cfg.sample_interval_s, prev=prev,
                )
            else:
                self._readers[src.name] = ObservationCollector._make_empty_reader(
                    src.dim_per_sub
                )

    # ── Gym 接口 ───────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._ensure_env()
        self._step_count = 0

        # 与数据采集一致：mj_resetData → 机械臂到位 → 物体随机化 → forward
        self._reset_mgr.reset(randomize_objects=True)

        obs = self._collect_obs()
        return obs, {}

    def step(
        self, action: np.ndarray,
    ) -> tuple[dict, float, bool, bool, dict]:
        self._ensure_env()
        self._apply_action(action)
        obs = self._step_physics_and_collect()
        self._step_count += 1

        terminated = bool(self._check_success())
        truncated = (
            self._max_episode_steps is not None
            and self._step_count >= self._max_episode_steps
        )
        reward = 1.0 if terminated else 0.0
        info = {"is_success": terminated}
        return obs, reward, terminated, truncated, info

    # 渲染 eval 视频回调
    def render(self) -> np.ndarray | None:
        self._ensure_env()
        if self._render_mode == "human":
            self._mj.sync_viewer()
            return None
        # 用场景 view 配置的自由视角渲染 RGB (H, W, 3)，供 eval 录制视频使用
        return self._renderer.render_view(self._scene_cfg.view)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._mj is not None:
            self._mj.close()
            self._mj = None

    # ── 属性 ───────────────────────────────────────────

    @property
    def task(self) -> str:
        return self._task_name

    @property
    def task_description(self) -> str:
        return self._task_name.replace("_", " ").title()

    # ── 内部：多速率采集 ───────────────────────────────

    def _step_physics_and_collect(self) -> dict:
        """推进一个记录帧，期间多速率采样状态 + 渲染相机。"""
        dc = self._dataset_cfg
        frame_dt = dc.frame_interval_s
        n_sub = max(1, dc.max_scale)
        sub_dt = frame_dt / n_sub
        n_phys_per_sub = max(1, round(sub_dt / self._mj.physics_dt))

        state_sources = dc.state_sources()
        samples: dict[str, list[np.ndarray]] = {s.name: [] for s in state_sources}

        for _ in range(n_sub):
            for _ in range(n_phys_per_sub):
                self._mj.step()
            for s in state_sources:
                samples[s.name].append(self._readers[s.name](self._mj))

        # 渲染相机
        frames = self._renderer.render_all(self._mj.data)

        obs: dict[str, np.ndarray] = {}
        if dc.flat_mode:
            parts = [
                np.stack(samples[s.name]).astype(np.float32)[0]
                for s in state_sources
            ]
            obs["state"] = np.concatenate(parts)
        else:
            for s in state_sources:
                obs[s.name] = np.stack(samples[s.name]).astype(np.float32)
        for f in frames:
            obs[f"images.{f.name}.rgb"] = np.ascontiguousarray(
                f.rgb.transpose(2, 0, 1)
            )
            obs[f"images.{f.name}.depth"] = f.depth[np.newaxis, ...]
        return obs

    def _collect_obs(self) -> dict:
        """reset 后采集初始观测（状态重复 R 次，相机渲染一次）。"""
        dc = self._dataset_cfg
        state_sources = dc.state_sources()
        obs: dict[str, np.ndarray] = {}
        if dc.flat_mode:
            parts = [
                self._readers[s.name](self._mj).astype(np.float32)
                for s in state_sources
            ]
            obs["state"] = np.concatenate(parts)
        else:
            for s in state_sources:
                val = self._readers[s.name](self._mj).astype(np.float32)
                obs[s.name] = np.repeat(val[np.newaxis, :], s.num_subs, axis=0)

        frames = self._renderer.render_all(self._mj.data)
        for f in frames:
            obs[f"images.{f.name}.rgb"] = np.ascontiguousarray(
                f.rgb.transpose(2, 0, 1)
            )
            obs[f"images.{f.name}.depth"] = f.depth[np.newaxis, ...]
        return obs

    def _apply_action(self, action: np.ndarray) -> None:
        apply_arm_action(
            self._mj, self._scene_cfg.robots,
            self._arm_act_ids, self._grip_act_ids, action,
        )

    def _init_teacher(self) -> None:
        """实例化任务对应 teacher，用于评估成功判定（自适应任务）。"""
        teacher_cfg_path = self._scene_cfg.task.teacher_config_file
        try:
            teacher_cfg = load_teacher_config(teacher_cfg_path)
        except Exception:
            log.warning(
                "无法加载 teacher 配置，成功检测不可用: %s", teacher_cfg_path,
            )
            self._teacher = None
            return

        ttype = getattr(teacher_cfg, "type", "")
        if ttype not in TEACHER_REGISTRY:
            log.warning(
                "teacher 类型 %r 未注册，成功检测不可用", ttype,
            )
            self._teacher = None
            return

        is_multi_arm = bool(
            getattr(TEACHER_REGISTRY[ttype], "_is_multi_arm", False)
        )
        if is_multi_arm:
            # 单实例控制多臂
            self._teacher = create_teacher(
                ttype, self._mj.model, self._mj.data, config=teacher_cfg,
            )
        else:
            # 每臂一个实例；成功判定为所有臂均成功
            self._teacher = MultiTeacherWrapper()
            for r in self._scene_cfg.robots:
                t = create_teacher(
                    ttype, self._mj.model, self._mj.data,
                    config=teacher_cfg, prefix=r.prefix,
                )
                self._teacher.add(t)

    def _check_success(self) -> bool:
        """成功检测 — 由任务对应 teacher 的 check_success() 判定（自适应）。"""
        if self._teacher is None:
            return False
        return bool(self._teacher.check_success())
