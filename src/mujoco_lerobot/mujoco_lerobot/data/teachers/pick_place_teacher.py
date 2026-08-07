"""Pick-and-Place Scripted Teacher — 含姿态检测和方向调整。

状态机:
  0. MOVE_ABOVE    — 移动到物块上方，计算最佳夹取角
  1. ROTATE        — 旋转夹爪到最佳方向（保持在上方）
  2. DESCEND       — 下降抓取
  3. CLOSE         — 闭合夹爪 + 抓取检测
  4. LIFT          — 抬起 + 掉落检测
  5. MOVE_TO_PLATE — 移动到盘子 + 掉落检测
  6. PLACE         — 下降放置
  7. OPEN          — 释放
  8. RETREAT       — 撤退 + 放置检测
  9. SUCCESS

输出绝对目标位姿 [x,y,z, qw,qx,qy,qz, gripper_cmd]。
所有阈值/高度/夹爪指令均从 configs/teachers/pick_place.yaml 读取。
"""

from __future__ import annotations

from enum import Enum

import mujoco
import numpy as np

from ...configs.teacher_config import PickPlaceTeacherConfig
from .base import Teacher, TeacherState
from . import register_teacher


class PickPlaceState(Enum):
    MOVE_ABOVE = 0
    ROTATE = 1
    DESCEND = 2
    CLOSE = 3
    LIFT = 4
    MOVE_TO_PLATE = 5
    PLACE = 6
    OPEN = 7
    RETREAT = 8
    GO_BACK = 9
    SUCCESS = 10


@register_teacher("PickPlaceTeacher")
class PickPlaceTeacher(Teacher):
    teacher_type = "PickPlaceTeacher"
    config_class = PickPlaceTeacherConfig

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: PickPlaceTeacherConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(model, data, config=config, prefix=prefix)
        cfg = config or PickPlaceTeacherConfig()
        self._cfg = cfg
        self.prefix = prefix
        self.cube_name = cfg.cube
        self.plate_name = cfg.plate
        self.grasp_quat = np.asarray(cfg.grasp_quat, dtype=np.float64)

        self.phase = PickPlaceState.MOVE_ABOVE
        self.phase_step = 0
        self._init_pose = self.get_ee_pose(self.prefixed_ee_site())
        self._target_pos = np.zeros(3)
        self._target_quat = self.grasp_quat.copy()
        self._grasp_yaw = 0.0
        self._retry_count = 0

    def prefixed_ee_site(self) -> str:
        return f"{self.prefix}_tcp" if self.prefix else "_tcp"

    def prefixed_object(self, name: str) -> str:
        return f"{self.prefix}{name}"

    # ── 生命周期 ───────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self._init_pose = self.get_ee_pose(self.prefixed_ee_site())
        self.phase = PickPlaceState.MOVE_ABOVE
        self.phase_step = 0
        self._retry_count = 0

    def step(self) -> dict[str, np.ndarray]:
        self.current_step += 1
        self.phase_step += 1
        try:
            return {
                PickPlaceState.MOVE_ABOVE: self._move_above,
                PickPlaceState.ROTATE: self._rotate,
                PickPlaceState.DESCEND: self._descend,
                PickPlaceState.CLOSE: self._close,
                PickPlaceState.LIFT: self._lift,
                PickPlaceState.MOVE_TO_PLATE: self._move_to_plate,
                PickPlaceState.PLACE: self._place,
                PickPlaceState.OPEN: self._open,
                PickPlaceState.RETREAT: self._retreat,
                PickPlaceState.GO_BACK: self._go_back,
                PickPlaceState.SUCCESS: self._success,
            }[self.phase]()
        except Exception:
            self.state = TeacherState.FAILURE
            return {self.prefix: self.make_action(np.zeros(3))}

    # ── MOVE_ABOVE ────────────────────────────────────

    def _move_above(self) -> np.ndarray:
        if self.phase_step == 1:
            cube = self.get_object_pose(self.prefixed_object(self.cube_name))
            self._target_pos = cube[:3] + np.array([0, 0, self._cfg.heights.above])

        ee = self.get_ee_pose(self.prefixed_ee_site())
        if np.linalg.norm(self._target_pos - ee[:3]) < self._cfg.thresh.approach_pos:
            self.phase = PickPlaceState.DESCEND
            self.phase_step = 0
        return {self.prefix: self.make_action(self._target_pos, self._target_quat)}

    # ── ROTATE ───────────────────────────────────────

    def _rotate(self) -> np.ndarray:
        if self.phase_step == 1:
            ee = self.get_ee_pose(self.prefixed_ee_site())
            self._target_pos = ee[:3].copy()
            cube = self.get_object_pose(self.prefixed_object(self.cube_name))
            self._grasp_yaw = self._compute_grasp_yaw(cube[3:])
            self._target_quat = self._make_grasp_quat(self._grasp_yaw)

        ee = self.get_ee_pose(self.prefixed_ee_site())
        err = np.zeros(3)
        mujoco.mju_subQuat(err, self._target_quat, ee[3:7])
        ang = np.linalg.norm(err)
        if ang < self._cfg.thresh.approach_rot:
            self.phase = PickPlaceState.DESCEND
            self.phase_step = 0
        return {self.prefix: self.make_action(self._target_pos, self._target_quat)}

    # ── DESCEND ───────────────────────────────────────

    def _descend(self) -> np.ndarray:
        if self.phase_step == 1:
            cube = self.get_object_pose(self.prefixed_object(self.cube_name))
            self._target_pos = cube[:3].copy()
            self._target_pos[2] += 0.01
        ee = self.get_ee_pose(self.prefixed_ee_site())
        if np.linalg.norm(self._target_pos - ee[:3]) < 0.003:
            self.phase = PickPlaceState.CLOSE
            self.phase_step = 0
        return {self.prefix: self.make_action(self._target_pos, self._target_quat)}

    # ── CLOSE ────────────────────────────────────────

    def _close(self) -> np.ndarray:
        ee = self.get_ee_pose(self.prefixed_ee_site())
        if self.phase_step > self._cfg.thresh.gripper_wait:
            cube = self.get_object_pose(self.prefixed_object(self.cube_name))
            if np.linalg.norm(cube[:3] - ee[:3]) < self._cfg.thresh.grasp_dist:
                self.phase = PickPlaceState.LIFT
                self.phase_step = 0
            else:
                return self._retry()
        return {
            self.prefix: self.make_action(
                ee[:3], ee[3:7], gripper_cmd=self._cfg.gripper.close,
            )
        }

    # ── LIFT ─────────────────────────────────────────

    def _lift(self) -> np.ndarray:
        ee = self.get_ee_pose(self.prefixed_ee_site())
        if self.phase_step > 1:
            cube = self.get_object_pose(self.prefixed_object(self.cube_name))
            if np.linalg.norm(cube[:3] - ee[:3]) > self._cfg.thresh.drop_dist:
                return self._retry()

        if self.phase_step == 1:
            self._target_pos = ee[:3] + np.array([0, 0, self._cfg.heights.lift])
            self._target_quat = ee[3:7]

        if np.linalg.norm(self._target_pos - ee[:3]) < self._cfg.thresh.approach_pos:
            self.phase = PickPlaceState.MOVE_TO_PLATE
            self.phase_step = 0
        return {
            self.prefix: self.make_action(
                self._target_pos, self._target_quat, gripper_cmd=self._cfg.gripper.close,
            )
        }

    # ── MOVE_TO_PLATE ────────────────────────────────

    def _move_to_plate(self) -> np.ndarray:
        ee = self.get_ee_pose(self.prefixed_ee_site())
        if self.phase_step > 1:
            cube = self.get_object_pose(self.prefixed_object(self.cube_name))
            if np.linalg.norm(cube[:3] - ee[:3]) > self._cfg.thresh.drop_dist:
                return self._retry()
        if self.phase_step == 1:
            self._target_quat = self.grasp_quat.copy()

        plate = self.get_object_pose(self.prefixed_object(self.plate_name))
        target = plate[:3] + np.array([0, 0, self._cfg.heights.above])
        if np.linalg.norm(target - ee[:3]) < self._cfg.thresh.approach_pos:
            self.phase = PickPlaceState.PLACE
            self.phase_step = 0
        return {
            self.prefix: self.make_action(
                target, self._target_quat, gripper_cmd=self._cfg.gripper.close,
            )
        }

    # ── PLACE ────────────────────────────────────────

    def _place(self) -> np.ndarray:
        ee = self.get_ee_pose(self.prefixed_ee_site())
        if self.phase_step > 1:
            cube = self.get_object_pose(self.prefixed_object(self.cube_name))
            if np.linalg.norm(cube[:3] - ee[:3]) > self._cfg.thresh.drop_dist:
                return self._retry()
        if self.phase_step == 1:
            self._target_quat = self.grasp_quat.copy()

        plate = self.get_object_pose(self.prefixed_object(self.plate_name))
        target = plate[:3] + np.array([0, 0, self._cfg.heights.place])
        if np.linalg.norm(target - ee[:3]) < 0.01:
            self.phase = PickPlaceState.OPEN
            self.phase_step = 0
        return {
            self.prefix: self.make_action(
                target, self._target_quat, gripper_cmd=self._cfg.gripper.close,
            )
        }

    # ── OPEN ─────────────────────────────────────────

    def _open(self) -> np.ndarray:
        ee = self.get_ee_pose(self.prefixed_ee_site())
        if self.phase_step > self._cfg.thresh.gripper_wait:
            self.phase = PickPlaceState.RETREAT
            self.phase_step = 0
        return {
            self.prefix: self.make_action(
                ee[:3], ee[3:7], gripper_cmd=self._cfg.gripper.open,
            )
        }

    # ── RETREAT ──────────────────────────────────────

    def _retreat(self) -> np.ndarray:
        ee = self.get_ee_pose(self.prefixed_ee_site())
        if self.phase_step == 1:
            self._target_pos = ee[:3] + np.array([0, 0, self._cfg.heights.retreat])
            self._target_quat = self.grasp_quat.copy()

        if np.linalg.norm(self._target_pos - ee[:3]) < self._cfg.thresh.approach_pos:
            if self.phase_step < self._cfg.thresh.settle_wait:
                return {
                    self.prefix: self.make_action(
                        self._target_pos, ee[3:7], gripper_cmd=self._cfg.gripper.open,
                    )
                }
            cube = self.get_object_pose(self.prefixed_object(self.cube_name))
            plate = self.get_object_pose(self.prefixed_object(self.plate_name))
            if np.linalg.norm(cube[:2] - plate[:2]) < self._cfg.thresh.place_dist:
                self.phase = PickPlaceState.GO_BACK
                self.phase_step = 0
            else:
                return self._retry()
        return {
            self.prefix: self.make_action(
                self._target_pos, self._target_quat, gripper_cmd=self._cfg.gripper.open,
            )
        }

    # ── GO_BACK ──────────────────────────────────────

    def _go_back(self) -> np.ndarray:
        ee = self.get_ee_pose(self.prefixed_ee_site())
        if self.phase_step == 1:
            self._target_pos = self._init_pose[:3].copy()
            self._target_quat = self._init_pose[3:7].copy()

        if np.linalg.norm(self._target_pos - ee[:3]) < self._cfg.thresh.approach_pos:
            if self.phase_step < self._cfg.thresh.settle_wait:
                return {
                    self.prefix: self.make_action(
                        self._target_pos, ee[3:7], gripper_cmd=self._cfg.gripper.open,
                    )
                }
            cube = self.get_object_pose(self.prefixed_object(self.cube_name))
            plate = self.get_object_pose(self.prefixed_object(self.plate_name))
            if np.linalg.norm(cube[:2] - plate[:2]) < self._cfg.thresh.place_dist:
                self.phase = PickPlaceState.SUCCESS
                self.phase_step = 0
            else:
                return self._retry()
        return {
            self.prefix: self.make_action(
                self._target_pos, self._target_quat, gripper_cmd=self._cfg.gripper.open,
            )
        }

    # ── SUCCESS ──────────────────────────────────────

    def _success(self) -> np.ndarray:
        self.state = TeacherState.SUCCESS
        return {self.prefix: self.make_action(np.zeros(3))}

    # ── 评估成功判定（不依赖状态机） ──────────────────

    def check_success(self) -> bool:
        """成功 = cube 已被放到 plate 上：cube 与 plate 中心 xy 距离 < place_dist。"""
        cube = self.get_object_pose(self.prefixed_object(self.cube_name))
        plate = self.get_object_pose(self.prefixed_object(self.plate_name))
        is_cube_in_plate = np.linalg.norm(cube[:2] - plate[:2]) < self._cfg.thresh.place_dist
        tcp = self.get_ee_pose(self.prefixed_ee_site())
        is_tcp_lifted = np.linalg.norm(tcp[:3] - plate[:3]) > 0.30
        is_success = is_cube_in_plate and is_tcp_lifted
        return is_success

    # ── 夹取姿态 ─────────────────────────────────────

    def _compute_grasp_yaw(self, cube_quat: np.ndarray) -> float:
        axes = self._cube_axes(cube_quat)
        z_dots = [abs(np.dot(a, [0, 0, 1])) for a in axes]
        vertical_idx = int(np.argmax(z_dots))

        candidates: list[float] = []
        for i in range(3):
            if i == vertical_idx:
                continue
            n = axes[i]
            if abs(n[2]) > 0.7:
                continue
            for sign in [1.0, -1.0]:
                nx, ny = sign * n[0], sign * n[1]
                if abs(nx) + abs(ny) < 0.01:
                    continue
                candidates.append(float(np.arctan2(ny, nx)))

        if not candidates:
            return 0.0

        ee = self.get_ee_pose(self.prefixed_ee_site())
        cy = self._quat_to_yaw(ee[3:])
        best = min(candidates, key=lambda t: min(
            abs(t - cy), abs(t - cy + 2 * np.pi), abs(t - cy - 2 * np.pi)))
        return best

    @staticmethod
    def _cube_axes(quat: np.ndarray) -> list[np.ndarray]:
        w, x, y, z = quat
        R = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ])
        return [R[:, 0], R[:, 1], R[:, 2]]

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        w, x, y, z = quat
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _make_grasp_quat(self, yaw: float) -> np.ndarray:
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        a = np.array([cy, 0, 0, sy])
        b = self.grasp_quat
        return np.array([
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ])

    def _retry(self) -> np.ndarray:
        self._retry_count += 1
        if self._retry_count > self._cfg.thresh.max_retries:
            self.state = TeacherState.FAILURE
            return {self.prefix: self.make_action(np.zeros(3))}
        self.phase = PickPlaceState.MOVE_ABOVE
        self.phase_step = 0
        return {self.prefix: self.make_action(np.zeros(3))}
