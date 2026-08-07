"""多速率观测采集器 — 支持嵌套 recode_scale 缓冲采样。

sample()    — 每个子采样间隔调用，把各数据源写入预分配的环形缓冲槽
is_ready()  — 缓冲区已满（max_scale 个子采样都已记录）
flush()     — 帧边界调用：注入最新相机帧 → 打包缓冲 → 清空 → 返回帧 dict

内存优化：
  - 每个数据源预分配 (num_subs, dim_per_sub) float32 缓冲，sample 直接写入槽位，
    避免「list 收集 + stack」导致的持续内存增长。
  - 关节/传感器读取使用 numpy 索引（mv：data.qpos[adrs] / sensordata[adr:adr+dim]）。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..configs.dataset_config import DatasetConfig
from ..simulate.mujoco_wrapper import MujocoWrapper

log = logging.getLogger(__name__)


def _quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """四元数 [w,x,y,z] → 欧拉角 [rx, ry, rz]（XYZ 顺序）。

    对应 R = Rz(yaw)·Ry(pitch)·Rx(roll) 的标准分解。
    """
    w, x, y, z = q
    # roll (绕 X)
    rx = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    # pitch (绕 Y)
    sp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    ry = np.arcsin(sp)
    # yaw (绕 Z)
    rz = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([rx, ry, rz])


class ObservationCollector:
    def __init__(
        self,
        mj: MujocoWrapper,
        dataset_cfg: DatasetConfig,
    ) -> None:
        self._mj = mj
        self._cfg = dataset_cfg

        # 校验名称（警告列表）
        for w in dataset_cfg.validate_names(mj.model):
            log.warning(w)

        self._sub_step = 0
        # 每源预分配缓冲: name -> (num_subs, dim) float32
        self._buffers: dict[str, np.ndarray] = {}
        # 每源读取器: name -> callable(mj) -> (dim,) float32
        self._readers: dict[str, callable] = {}
        # 每源采样周期（在 max_scale 网格中的分布）
        self._periods: dict[str, int] = {}
        # 每源已采样的槽计数
        self._counts: dict[str, int] = {}
        # 最近动作（action 源数据）
        total_action_dim = sum(s.dim_per_sub for s in dataset_cfg.action_sources())
        self._last_action = np.zeros(total_action_dim, dtype=np.float32)
        self._action_offsets: dict[str, int] = {}
        # 相机最新帧: name -> {"rgb":..., "depth":...}（跨记录帧复用，reset 时清空）
        self._camera_frame: dict[str, dict[str, np.ndarray]] = {}
        # site 位姿 relative/velocity 类型的上一采样值
        self._pose_prev: dict[str, np.ndarray] = {}
        # 采样间隔（velocity 类型除以 dt）
        self._sample_dt = dataset_cfg.sample_interval_s

        self._init_readers()

    # ── 初始化读取器 ────────────────────────────────────

    def _init_readers(self) -> None:
        mj = self._mj
        max_scale = self._cfg.max_scale
        action_offset = 0
        for src in self._cfg.sources:
            # 关节类 action 源从 _last_action 读取；site 类（tcp）从实时位姿读取
            if src.name.startswith("action.") and not src.source_type.startswith("site_"):
                self._action_offsets[src.name] = action_offset
                self._readers[src.name] = self._make_action_reader(src, action_offset)
                action_offset += src.dim_per_sub
            elif src.source_type == "joint_pos":
                adrs = [mj._jnt_qposadr.get(n, -1) for n in src.read_names]
                self._readers[src.name] = self._make_qpos_reader(adrs, src.dim_per_sub)
            elif src.source_type == "joint_vel":
                adrs = [mj._jnt_dofadr.get(n, -1) for n in src.read_names]
                self._readers[src.name] = self._make_qvel_reader(adrs, src.dim_per_sub)
            elif src.source_type == "joint_effort":
                adrs = [mj._jnt_dofadr.get(n, -1) for n in src.read_names]
                self._readers[src.name] = self._make_effort_reader(adrs, src.dim_per_sub)
            elif src.source_type.startswith("sensor."):
                slices = [mj.sensor_slice(n) for n in src.read_names]
                self._readers[src.name] = self._make_sensor_reader(slices, src.dim_per_sub)
            elif src.source_type.startswith("site_"):
                suffix = src.source_type.rsplit("_", 1)[-1]  # position / euler / quat
                sids = [mj._site_ids.get(n, -1) for n in src.read_names]
                fid = mj._site_ids.get(src.frame_site) if src.frame_site != "world" else None
                per = 3 if suffix != "quat" else 4
                prev = np.zeros(len(sids) * per, dtype=np.float32)
                self._pose_prev[src.name] = prev
                self._readers[src.name] = self._make_site_reader(
                    sids, suffix, src.dim_per_sub, fid,
                    src.pose_type, self._sample_dt, prev=prev,
                )
            else:
                self._readers[src.name] = self._make_empty_reader(src.dim_per_sub)

            num_subs = max(1, src.num_subs)
            self._buffers[src.name] = np.zeros((num_subs, src.dim_per_sub), dtype=np.float32)
            # 采样周期：把 num_subs 均匀分布到 max_scale 网格
            self._periods[src.name] = max(1, (max_scale + num_subs - 1) // num_subs)
            self._counts[src.name] = 0

    def _make_action_reader(self, src, offset: int):
        dim = src.dim_per_sub

        def _read(mj: MujocoWrapper) -> np.ndarray:
            return self._last_action[offset : offset + dim]

        return _read

    @staticmethod
    def _make_qpos_reader(adrs: list[int], dim: int):
        valid = [(i, a) for i, a in enumerate(adrs) if a >= 0]

        def _read(mj: MujocoWrapper) -> np.ndarray:
            out = np.zeros(dim, dtype=np.float32)
            qpos = mj.data.qpos
            for i, a in valid:
                out[i] = qpos[a]
            return out

        return _read

    @staticmethod
    def _make_qvel_reader(adrs: list[int], dim: int):
        """关节速度读取器（data.qvel[dofadr]）。"""
        valid = [(i, a) for i, a in enumerate(adrs) if a >= 0]

        def _read(mj: MujocoWrapper) -> np.ndarray:
            out = np.zeros(dim, dtype=np.float32)
            qvel = mj.data.qvel
            for i, a in valid:
                out[i] = qvel[a]
            return out

        return _read

    @staticmethod
    def _make_effort_reader(adrs: list[int], dim: int):
        """关节力矩读取器（data.qfrc_actuator[dofadr]，广义执行器力）。"""
        valid = [(i, a) for i, a in enumerate(adrs) if a >= 0]

        def _read(mj: MujocoWrapper) -> np.ndarray:
            out = np.zeros(dim, dtype=np.float32)
            qfrc = mj.data.qfrc_actuator
            for i, a in valid:
                out[i] = qfrc[a]
            return out

        return _read

    @staticmethod
    def _make_sensor_reader(slices: list[tuple[int, int]], dim: int):
        valid = [
            (i, sadr, min(sdim, 3))
            for i, (sadr, sdim) in enumerate(slices) if sadr >= 0
        ]

        def _read(mj: MujocoWrapper) -> np.ndarray:
            out = np.zeros(dim, dtype=np.float32)
            sdata = mj.data.sensordata
            for i, sadr, sdim in valid:
                out[i * 3 : i * 3 + sdim] = sdata[sadr : sadr + sdim]
            return out

        return _read

    @staticmethod
    def _make_site_reader(
        sids: list[int],
        suffix: str,
        dim: int,
        frame_site_id: int | None,
        pose_type: str,
        sample_dt: float,
        prev: np.ndarray | None = None,
    ):
        """site 位姿读取器（position/euler/quat，支持参考系与 relative/velocity）。

        - frame_site_id=None 表示 world 参考系。
        - pose_type=relative 记录相邻采样的差值；velocity 记录差值 / 采样间隔。
        - prev：可选的上一采样值缓冲（由调用方持有，供 reset 复用）。
        """
        import mujoco

        n = len(sids)
        valid = [(i, s) for i, s in enumerate(sids) if s >= 0]
        per = 3 if suffix != "quat" else 4
        if prev is None:
            prev = np.zeros(n * per, dtype=np.float32)

        def _read(mj: MujocoWrapper) -> np.ndarray:
            out = np.zeros(n * per, dtype=np.float32)
            xpos = mj.data.site_xpos
            xmat = mj.data.site_xmat   # 每个 site 一个 9 元素行主序旋转矩阵 (nsite, 9)

            if suffix == "position":
                for i, s in valid:
                    out[i * 3 : i * 3 + 3] = xpos[s]
                if frame_site_id is not None:
                    fpos = xpos[frame_site_id]
                    rmat = xmat[frame_site_id].reshape(3, 3)  # R_frame
                    for i in range(n):
                        p = out[i * 3 : i * 3 + 3] - fpos
                        out[i * 3 : i * 3 + 3] = rmat.T @ p
            else:
                # quat / euler：先取旋转矩阵，再按参考系旋转（R_rel = R_frame^T · R_site）
                qbuf = np.empty(4)
                rmat = (
                    xmat[frame_site_id].reshape(3, 3)
                    if frame_site_id is not None
                    else None
                )
                for i, s in valid:
                    mat = xmat[s]  # (9,) 行主序
                    if rmat is not None:
                        rel = rmat.T @ mat.reshape(3, 3)
                        mat = rel.reshape(-1)
                    mujoco.mju_mat2Quat(qbuf, mat)
                    if suffix == "euler":
                        out[i * 3 : i * 3 + 3] = _quat_to_euler_xyz(qbuf)
                    else:
                        out[i * 4 : i * 4 + 4] = qbuf

            if pose_type == "relative":
                delta = out - prev
                prev[:] = out
                return delta
            if pose_type == "velocity":
                vel = (out - prev) / sample_dt
                prev[:] = out
                return vel
            return out

        return _read

    @staticmethod
    def _make_empty_reader(dim: int):
        def _read(mj: MujocoWrapper) -> np.ndarray:
            return np.zeros(dim, dtype=np.float32)

        return _read

    # ── 生命周期 ───────────────────────────────────────

    def reset(self) -> None:
        self._sub_step = 0
        self._last_action.fill(0.0)
        for name in self._buffers:
            self._buffers[name].fill(0.0)
            self._counts[name] = 0
        for name in self._pose_prev:
            self._pose_prev[name].fill(0.0)
        self._camera_frame.clear()

    def update_last_action(self, action: np.ndarray) -> None:
        """记录最近一次动作，供 action 数据源采样。"""
        arr = np.asarray(action, dtype=np.float32).ravel()
        n = min(len(arr), len(self._last_action))
        self._last_action[:n] = arr[:n]

    # ── 采样与刷新 ─────────────────────────────────────

    def sample(self) -> None:
        """在当前子步记录所有应采样的数据源。"""
        mj = self._mj
        for name, src in self._iter_sources():
            period = self._periods[name]
            if self._sub_step % period == 0 and self._counts[name] < src.num_subs:
                slot = self._counts[name]
                self._buffers[name][slot] = self._readers[name](mj)
                self._counts[name] = slot + 1
        self._sub_step += 1

    def _iter_sources(self):
        for src in self._cfg.sources:
            yield src.name, src

    def is_ready(self) -> bool:
        return self._sub_step >= self._cfg.max_scale

    def set_camera_frames(self, frames: dict[str, dict[str, np.ndarray]]) -> None:
        """注入最新相机帧（由 SimulationManager 渲染后调用）。"""
        self._camera_frame.update(frames)

    def flush(self, task_id: int) -> dict[str, Any]:
        """打包缓冲区 + 最新相机帧 → 返回帧 dict。

        相机帧跨记录帧复用：相机渲染频率 = 场景配置 fps，记录频率 =
        recode_hz；两次相机渲染之间的记录帧复用最近一帧相机图像。
        """
        state: dict[str, np.ndarray] = {}
        action: dict[str, np.ndarray] = {}
        for src in self._cfg.sources:
            buf = self._buffers[src.name]
            n = self._counts[src.name]
            # 必须拷贝：随后会 fill(0.0) 清空缓冲，若留视图会把已组帧数据一并清零
            data = buf[:n].copy()
            if src.name.startswith("state."):
                state[src.name] = data
            else:
                action[src.name] = data

        images = dict(self._camera_frame)

        # 清空状态缓冲（相机帧保留，供下一记录帧复用）
        self._sub_step = 0
        for name in self._buffers:
            self._buffers[name].fill(0.0)
            self._counts[name] = 0

        return {"state": state, "action": action, "images": images, "task_id": int(task_id)}
