"""MuJoCo 底层封装 — 纯物理 API + 高频读取优化。

设计要点（性能/内存）：
- `_ensure_env()` 延迟初始化，兼容 gym.vector.AsyncVectorEnv 的 fork 机制
- 预计算 joint → qposadr / dofadr、actuator id、sensor 切片，避免每步 Python 循环
- 批量关节读取使用 numpy 花式索引（`data.qpos[adrs]`），零逐关节循环
- 传感器读取返回 `sensordata` 切片视图（不拷贝），由调用方决定是否复制
- viewer 延迟启动（render=True 自动启动，或手动 launch_viewer）
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

try:  # viewer 为可选模块
    import mujoco.viewer  # noqa: F401
except Exception:  # pragma: no cover
    pass


class MujocoWrapper:
    def __init__(
        self,
        scene_path: str | Path,
        render: bool = False,
    ) -> None:
        self._scene_path = Path(scene_path)
        self._render_flag = render

        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.viewer: Optional[mujoco.viewer.Handle] = None

        # 预计算缓存（_ensure_env 时填充）
        self._jnt_qposadr: dict[str, int] = {}
        self._jnt_dofadr: dict[str, int] = {}
        self._actuator_ids: dict[str, int] = {}
        self._sensor_slices: dict[str, tuple[int, int]] = {}
        self._body_ids: dict[str, int] = {}
        self._site_ids: dict[str, int] = {}

    # ── 生命周期 ───────────────────────────────────────

    def open(self) -> None:
        self._ensure_env()

    def _ensure_env(self) -> None:
        if self.model is not None:
            return
        if not self._scene_path.exists():
            raise FileNotFoundError(f"mjcf 场景不存在: {self._scene_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self._scene_path))
        self.data = mujoco.MjData(self.model)

        # 预计算索引缓存
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            self._jnt_qposadr[name] = int(self.model.jnt_qposadr[i])
            self._jnt_dofadr[name] = int(self.model.jnt_dofadr[i])
        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            self._actuator_ids[name] = i
        for i in range(self.model.nsensor):
            name = self.model.sensor(i).name
            self._sensor_slices[name] = (
                int(self.model.sensor_adr[i]),
                int(self.model.sensor_dim[i]),
            )
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name:
                self._body_ids[name] = i
        for i in range(self.model.nsite):
            name = self.model.site(i).name
            if name:
                self._site_ids[name] = i

        if self._render_flag:
            self.launch_viewer()

    def launch_viewer(self, key_callback=None) -> None:
        self._ensure_env()
        if self.viewer is not None:
            return
        try:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=key_callback
            )
        except Exception as exc:  # pragma: no cover - 无显示环境（如无头 CI/WSL）下失败
            import logging
            logging.getLogger(__name__).warning(
                "无法启动 MuJoCo viewer（可能无显示环境）: %s", exc
            )
            self.viewer = None

    def close(self) -> None:
        self.close_viewer()
        self.model = None
        self.data = None

    def close_viewer(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

    # ── 仿真控制 ───────────────────────────────────────

    def reset(self) -> None:
        self._ensure_env()
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        self._ensure_env()
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        self._ensure_env()
        mujoco.mj_forward(self.model, self.data)

    @property
    def physics_dt(self) -> float:
        self._ensure_env()
        return float(self.model.opt.timestep)

    @property
    def sim_time(self) -> float:
        self._ensure_env()
        return float(self.data.time)

    @property
    def camera_names(self) -> list[str]:
        self._ensure_env()
        return [self.model.camera(i).name for i in range(self.model.ncam)]

    # ── 状态读写 ───────────────────────────────────────

    def get_qpos(self) -> np.ndarray:
        self._ensure_env()
        return self.data.qpos.copy()

    def set_qpos(self, qpos: np.ndarray) -> None:
        self._ensure_env()
        self.data.qpos[:] = qpos

    def get_qvel(self) -> np.ndarray:
        self._ensure_env()
        return self.data.qvel.copy()

    def set_qvel(self, qvel: np.ndarray) -> None:
        self._ensure_env()
        self.data.qvel[:] = qvel

    def get_ctrl(self) -> np.ndarray:
        self._ensure_env()
        return self.data.ctrl.copy()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        self._ensure_env()
        self.data.ctrl[:] = ctrl

    # ── 关节查询 ───────────────────────────────────────

    def get_joint_id(self, name: str) -> int:
        self._ensure_env()
        return int(self.model.joint(name).id)

    def get_joint_qposadr(self, name: str) -> int:
        self._ensure_env()
        return self._jnt_qposadr[name]

    def get_actuator_id(self, name: str) -> int:
        self._ensure_env()
        return self._actuator_ids[name]

    def joint_qpos(self, joint_names: list[str]) -> np.ndarray:
        """批量读取关节位置（numpy 花式索引，无逐关节循环）。"""
        self._ensure_env()
        adrs = [self._jnt_qposadr[n] for n in joint_names]
        return self.data.qpos[adrs]

    def joint_qvel(self, joint_names: list[str]) -> np.ndarray:
        self._ensure_env()
        adrs = [self._jnt_dofadr[n] for n in joint_names]
        return self.data.qvel[adrs]

    def set_joint_qpos(self, name: str, value: float) -> None:
        self._ensure_env()
        self.data.qpos[self._jnt_qposadr[name]] = value

    # ── 传感器 ─────────────────────────────────────────

    def sensor_slice(self, name: str) -> tuple[int, int]:
        self._ensure_env()
        return self._sensor_slices[name]

    def get_sensor(self, name: str, *, copy: bool = False) -> np.ndarray:
        """读取传感器数据。copy=False 返回 sensordata 切片视图（零拷贝）。"""
        self._ensure_env()
        adr, dim = self._sensor_slices[name]
        out = self.data.sensordata[adr : adr + dim]
        return out.copy() if copy else out

    # ── body / site ────────────────────────────────────

    def get_body_pose(self, body_name: str) -> np.ndarray:
        """获取刚体位姿 [x,y,z, qw,qx,qy,qz]。"""
        self._ensure_env()
        bid = self._body_ids[body_name]
        pos = self.data.xpos[bid]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.xmat[bid])
        return np.concatenate([pos, quat])

    def get_site_pose(self, site_name: str) -> np.ndarray:
        """获取 site 位姿 [x,y,z, qw,qx,qy,qz]。"""
        self._ensure_env()
        sid = self._site_ids[site_name]
        pos = self.data.site_xpos[sid]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[sid])
        return np.concatenate([pos, quat])

    def get_body_joint_id(self, body_name: str) -> int | None:
        """获取物体 freejoint 的关节索引，无则返回 None。"""
        self._ensure_env()
        bid = self._body_ids.get(body_name)
        if bid is None:
            return None
        jntadr = self.model.body_jntadr[bid]
        if jntadr < 0 or self.model.body_jntnum[bid] == 0:
            return None
        return int(jntadr)

    def set_body_qpos(
        self, body_name: str, pos: np.ndarray, quat: np.ndarray | None = None,
    ) -> None:
        """设置 free body 的位姿（需 mj_forward 生效）。"""
        self._ensure_env()
        jntadr = self.model.body_jntadr[self._body_ids[body_name]]
        if jntadr < 0:
            return
        self.data.qpos[jntadr : jntadr + 3] = pos[:3]
        if quat is not None:
            self.data.qpos[jntadr + 3 : jntadr + 7] = quat[:4]

    # ── Viewer ─────────────────────────────────────────

    def is_viewer_running(self) -> bool:
        return self.viewer is not None and self.viewer.is_running()

    def sync_viewer(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()

    def set_viewer_camera(
        self,
        lookat: tuple[float, float, float] = (0.45, 0.0, 0.65),
        distance: float = 1.8,
        elevation: float = -25.0,
        azimuth: float = 130.0,
    ) -> None:
        if self.viewer is not None:
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = lookat
                self.viewer.cam.distance = distance
                self.viewer.cam.elevation = elevation
                self.viewer.cam.azimuth = azimuth
