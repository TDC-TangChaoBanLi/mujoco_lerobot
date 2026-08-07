"""双机械臂 Pick-and-Place Scripted Teacher — 交叉搬运任务。

一个 teacher 实例同时控制两个机械臂，各自运行独立状态机，
通过比较 _arm_a.phase / _arm_b.phase 实现同步。

Phase 1: 抓取自己的物块 → 放到桌面中心随机位置
Phase 2: 抓取对方的物块 → 放到对方的盘子上

输出绝对目标位姿 [x,y,z, qw,qx,qy,qz, gripper_cmd]。
所有阈值/高度/中心/避让参数均从 configs/teachers/dual_pick_place.yaml 读取。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import mujoco
import numpy as np

from ...configs.teacher_config import DualPickPlaceTeacherConfig
from .base import Teacher, TeacherState
from . import register_teacher


class DualPickPhase(Enum):
    MOVE_ABOVE_OWN = 0
    ROTATE_OWN = 1
    DESCEND_OWN = 2
    CLOSE_OWN = 3
    LIFT_OWN = 4
    MOVE_TO_CENTER = 5
    PLACE_AT_CENTER = 6
    OPEN_AT_CENTER = 7
    LIFT_FROM_CENTER = 8
    RETREAT_TO_SAFE = 9
    MOVE_ABOVE_OTHER = 10
    ROTATE_OTHER = 11
    DESCEND_OTHER = 12
    CLOSE_OTHER = 13
    LIFT_OTHER = 14
    MOVE_ABOVE_PLATE = 15
    PLACE_ON_PLATE = 16
    OPEN_ON_PLATE = 17
    LIFT_FROM_PLATE = 18
    RETREAT_HOME = 19
    SUCCESS = 20


@dataclass
class ArmState:
    """单个机械臂的不可变配置 + 可变状态。"""

    prefix: str
    ee_site: str
    grasp_euler: tuple[float, float, float]
    retreat_center: tuple[float, float]
    own_cube: str = ""
    own_plate: str = ""
    other_cube: str = ""
    target_plate: str = ""
    init_pose: np.ndarray = field(default_factory=lambda: np.zeros(7))
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_quat: np.ndarray = field(default_factory=lambda: np.zeros(4))
    grasp_yaw: float = 0.0
    gripper_cmd: float = 0.0
    phase: DualPickPhase = DualPickPhase.MOVE_ABOVE_OWN
    phase_step: int = 0
    retry_count: int = 0
    center_x: float = 0.0
    center_y: float = 0.0
    retreat_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    retreat_ready: bool = False
    retreat_sub_step: int = 0


@register_teacher("DualPickPlaceTeacher")
class DualPickPlaceTeacher(Teacher):
    """双机械臂 Scripted Teacher — 交叉搬运。"""

    _is_multi_arm = True
    teacher_type = "DualPickPlaceTeacher"
    config_class = DualPickPlaceTeacherConfig

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: DualPickPlaceTeacherConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(model, data, config=config, prefix=prefix)
        cfg = config or DualPickPlaceTeacherConfig()
        self._cfg = cfg
        self._arm_a = ArmState(
            prefix="A_", ee_site="A__tcp",
            grasp_euler=cfg.grasp_euler_a,
            retreat_center=cfg.retreat_center_a,
        )
        self._arm_b = ArmState(
            prefix="B_", ee_site="B__tcp",
            grasp_euler=cfg.grasp_euler_b,
            retreat_center=cfg.retreat_center_b,
        )
        self._arms = [self._arm_a, self._arm_b]
        for arm in self._arms:
            other = "B_" if arm.prefix.startswith("A") else "A_"
            arm.own_cube = f"{arm.prefix}cube"
            arm.own_plate = f"{arm.prefix}plate"
            arm.other_cube = f"{other}cube"
            arm.target_plate = f"{other}plate"
            arm.target_quat = self._euler_to_quat(*arm.grasp_euler)

    # ── 生命周期 ───────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        cy = float(np.random.uniform(*self._cfg.center_y_half_range))
        for arm in self._arms:
            arm.init_pose = self.get_ee_pose(arm.ee_site)
            arm.phase = DualPickPhase.MOVE_ABOVE_OWN
            arm.phase_step = 0
            arm.retry_count = 0
            arm.gripper_cmd = self._cfg.gripper.open
            arm.retreat_ready = False
            arm.retreat_sub_step = 0
            arm.center_x = float(np.random.uniform(*self._cfg.center_x_range))
            arm.center_y = cy if arm.prefix.startswith("A") else -cy
            self._randomize_retreat_pos(arm)

    def step(self) -> dict[str, np.ndarray]:
        self.current_step += 1
        result: dict[str, np.ndarray] = {}
        for arm in self._arms:
            arm.phase_step += 1
            try:
                pos, quat = self._dispatch(arm)
                result[arm.prefix] = self.make_action(pos, quat, arm.gripper_cmd)
            except Exception:
                self.state = TeacherState.FAILURE
                result[arm.prefix] = self.make_action(np.zeros(3))
        return result

    def _other(self, arm: ArmState) -> ArmState:
        return self._arm_b if arm.prefix.startswith("A") else self._arm_a

    # ── 状态分发 ───────────────────────────────────────

    def _dispatch(self, arm: ArmState) -> tuple[np.ndarray, np.ndarray]:
        return {
            DualPickPhase.MOVE_ABOVE_OWN: lambda: self._move_above_own(arm),
            DualPickPhase.ROTATE_OWN: lambda: self._rotate_own(arm),
            DualPickPhase.DESCEND_OWN: lambda: self._descend_own(arm),
            DualPickPhase.CLOSE_OWN: lambda: self._close_own(arm),
            DualPickPhase.LIFT_OWN: lambda: self._lift_own(arm),
            DualPickPhase.MOVE_TO_CENTER: lambda: self._move_to_center(arm),
            DualPickPhase.PLACE_AT_CENTER: lambda: self._place_at_center(arm),
            DualPickPhase.OPEN_AT_CENTER: lambda: self._open_at_center(arm),
            DualPickPhase.LIFT_FROM_CENTER: lambda: self._lift_from_center(arm),
            DualPickPhase.RETREAT_TO_SAFE: lambda: self._retreat_to_safe(arm),
            DualPickPhase.MOVE_ABOVE_OTHER: lambda: self._move_above_other(arm),
            DualPickPhase.ROTATE_OTHER: lambda: self._rotate_other(arm),
            DualPickPhase.DESCEND_OTHER: lambda: self._descend_other(arm),
            DualPickPhase.CLOSE_OTHER: lambda: self._close_other(arm),
            DualPickPhase.LIFT_OTHER: lambda: self._lift_other(arm),
            DualPickPhase.MOVE_ABOVE_PLATE: lambda: self._move_above_plate(arm),
            DualPickPhase.PLACE_ON_PLATE: lambda: self._place_on_plate(arm),
            DualPickPhase.OPEN_ON_PLATE: lambda: self._open_on_plate(arm),
            DualPickPhase.LIFT_FROM_PLATE: lambda: self._lift_from_plate(arm),
            DualPickPhase.RETREAT_HOME: lambda: self._retreat_home(arm),
            DualPickPhase.SUCCESS: lambda: self._success(arm),
        }[arm.phase]()

    # ══════════════════════ Phase 1 ══════════════════════

    def _move_above_own(self, arm: ArmState):
        return self._do_move_above(arm, arm.own_cube, DualPickPhase.ROTATE_OWN)

    def _rotate_own(self, arm: ArmState):
        return self._do_rotate_to_grasp(arm, arm.own_cube, DualPickPhase.DESCEND_OWN)

    def _descend_own(self, arm: ArmState):
        return self._do_descend(arm, arm.own_cube, DualPickPhase.CLOSE_OWN)

    def _close_own(self, arm: ArmState):
        return self._do_close_and_check(arm, arm.own_cube, DualPickPhase.LIFT_OWN)

    def _lift_own(self, arm: ArmState):
        return self._do_lift_with_check(
            arm, arm.own_cube, self._cfg.heights.lift, DualPickPhase.MOVE_TO_CENTER
        )

    def _move_to_center(self, arm: ArmState):
        ee = self.get_ee_pose(arm.ee_site)
        if arm.phase_step > 1 and self._object_far_from_ee(arm.own_cube, ee[:3]):
            return self._retry(arm)
        if arm.phase_step == 1:
            arm.target_pos = np.array([
                arm.center_x, arm.center_y,
                self._cfg.table_z + self._cfg.heights.above,
            ])
            arm.target_quat = self._euler_to_quat(*arm.grasp_euler)
        if self._arrived_at_pos(arm, self._cfg.thresh.approach_pos):
            self._transition(arm, DualPickPhase.PLACE_AT_CENTER)
        return arm.target_pos, arm.target_quat

    def _place_at_center(self, arm: ArmState):
        ee = self.get_ee_pose(arm.ee_site)
        if arm.phase_step > 1 and self._object_far_from_ee(arm.own_cube, ee[:3]):
            return self._retry(arm)
        if arm.phase_step == 1:
            arm.target_pos = np.array([
                arm.center_x, arm.center_y, self._cfg.table_z + 0.02,
            ])
            arm.target_quat = self._euler_to_quat(*arm.grasp_euler)
        if self._arrived_at_pos(arm, 0.01):
            self._transition(arm, DualPickPhase.OPEN_AT_CENTER)
        return arm.target_pos, arm.target_quat

    def _open_at_center(self, arm: ArmState):
        return self._do_open_and_wait(arm, DualPickPhase.LIFT_FROM_CENTER)

    def _lift_from_center(self, arm: ArmState):
        return self._do_lift_from_place(
            arm, self._cfg.heights.retreat, DualPickPhase.RETREAT_TO_SAFE
        )

    def _retreat_to_safe(self, arm: ArmState):
        return self._do_retreat_synced(
            arm, arm.retreat_pos, arm.own_cube,
            np.array([arm.center_x, arm.center_y]),
            DualPickPhase.MOVE_ABOVE_OTHER, DualPickPhase.RETREAT_TO_SAFE,
        )

    # ══════════════════════ Phase 2 ══════════════════════

    def _move_above_other(self, arm: ArmState):
        if self._other(arm).phase.value < DualPickPhase.RETREAT_TO_SAFE.value:
            return arm.target_pos, arm.target_quat
        return self._do_move_above(arm, arm.other_cube, DualPickPhase.ROTATE_OTHER)

    def _rotate_other(self, arm: ArmState):
        return self._do_rotate_to_grasp(arm, arm.other_cube, DualPickPhase.DESCEND_OTHER)

    def _descend_other(self, arm: ArmState):
        return self._do_descend(arm, arm.other_cube, DualPickPhase.CLOSE_OTHER)

    def _close_other(self, arm: ArmState):
        return self._do_close_and_check(arm, arm.other_cube, DualPickPhase.LIFT_OTHER)

    def _lift_other(self, arm: ArmState):
        return self._do_lift_with_check(
            arm, arm.other_cube, self._cfg.heights.lift, DualPickPhase.MOVE_ABOVE_PLATE
        )

    def _move_above_plate(self, arm: ArmState):
        ee = self.get_ee_pose(arm.ee_site)
        if arm.phase_step > 1 and self._object_far_from_ee(arm.other_cube, ee[:3]):
            return self._retry(arm)
        if arm.phase_step == 1:
            plate = self.get_object_pose(arm.target_plate)
            arm.target_pos = plate[:3] + np.array([0, 0, self._cfg.heights.above])
            arm.target_quat = self._euler_to_quat(*arm.grasp_euler)
        if self._arrived_at_pos(arm, self._cfg.thresh.approach_pos):
            self._transition(arm, DualPickPhase.PLACE_ON_PLATE)
        return arm.target_pos, arm.target_quat

    def _place_on_plate(self, arm: ArmState):
        ee = self.get_ee_pose(arm.ee_site)
        if arm.phase_step > 1 and self._object_far_from_ee(arm.other_cube, ee[:3]):
            return self._retry(arm)
        if arm.phase_step == 1:
            plate = self.get_object_pose(arm.target_plate)
            arm.target_pos = plate[:3] + np.array([0, 0, self._cfg.heights.place])
            arm.target_quat = self._euler_to_quat(*arm.grasp_euler)
        if self._arrived_at_pos(arm, 0.01):
            self._transition(arm, DualPickPhase.OPEN_ON_PLATE)
        return arm.target_pos, arm.target_quat

    def _open_on_plate(self, arm: ArmState):
        return self._do_open_and_wait(arm, DualPickPhase.LIFT_FROM_PLATE)

    def _lift_from_plate(self, arm: ArmState):
        return self._do_lift_from_place(
            arm, self._cfg.heights.retreat, DualPickPhase.RETREAT_HOME
        )

    def _retreat_home(self, arm: ArmState):
        plate = self.get_object_pose(arm.target_plate)
        return self._do_retreat_synced(
            arm, arm.init_pose[:3].copy(), arm.other_cube,
            plate[:2],
            DualPickPhase.SUCCESS, DualPickPhase.RETREAT_HOME,
        )

    def _success(self, arm: ArmState):
        if self._other(arm).phase == DualPickPhase.SUCCESS:
            self.state = TeacherState.SUCCESS
        return arm.target_pos, arm.target_quat

    # ══════════════════════ 操作原语 ══════════════════════

    def _do_move_above(self, arm: ArmState, obj_name: str, next_phase):
        if arm.phase_step == 1:
            obj = self.get_object_pose(obj_name)
            arm.target_pos = obj[:3] + np.array([0, 0, self._cfg.heights.above])
            arm.target_quat = self._euler_to_quat(*arm.grasp_euler)
            arm.gripper_cmd = self._cfg.gripper.open
        if self._arrived_at_pos(arm, self._cfg.thresh.approach_pos):
            self._transition(arm, next_phase)
        return arm.target_pos, arm.target_quat

    def _do_rotate_to_grasp(self, arm: ArmState, obj_name: str, next_phase):
        if arm.phase_step == 1:
            ee = self.get_ee_pose(arm.ee_site)
            arm.target_pos = ee[:3].copy()
            obj = self.get_object_pose(obj_name)
            arm.grasp_yaw = self._compute_grasp_yaw(arm, obj[3:])
            arm.target_quat = self._make_grasp_quat(arm.grasp_yaw)
        err = np.zeros(3)
        mujoco.mju_subQuat(
            err, arm.target_quat, self.get_ee_pose(arm.ee_site)[3:7]
        )
        if np.linalg.norm(err) < self._cfg.thresh.approach_rot:
            self._transition(arm, next_phase)
        return arm.target_pos, arm.target_quat

    def _do_descend(self, arm: ArmState, obj_name: str, next_phase):
        if arm.phase_step == 1:
            obj = self.get_object_pose(obj_name)
            arm.target_pos = obj[:3].copy()
            arm.target_pos[2] += 0.01
        if self._arrived_at_pos(arm, self._cfg.thresh.approach_pos):
            self._transition(arm, next_phase)
        return arm.target_pos, arm.target_quat

    def _do_close_and_check(self, arm: ArmState, obj_name: str, next_phase):
        ee = self.get_ee_pose(arm.ee_site)
        if arm.phase_step > self._cfg.thresh.gripper_wait:
            obj = self.get_object_pose(obj_name)
            if np.linalg.norm(obj[:3] - ee[:3]) < self._cfg.thresh.grasp_dist:
                arm.gripper_cmd = self._cfg.gripper.close
                self._transition(arm, next_phase)
            else:
                return self._retry(arm)
        if arm.phase_step == 1:
            arm.gripper_cmd = self._cfg.gripper.close
        return arm.target_pos, arm.target_quat

    def _do_lift_with_check(
        self, arm: ArmState, obj_name: str, height: float, next_phase
    ):
        ee = self.get_ee_pose(arm.ee_site)
        if arm.phase_step > 1 and self._object_far_from_ee(obj_name, ee[:3]):
            return self._retry(arm)
        if arm.phase_step == 1:
            arm.target_pos = ee[:3] + np.array([0, 0, height])
            arm.target_quat = ee[3:7].copy()
        if self._arrived_at_pos(arm, self._cfg.thresh.approach_pos):
            self._transition(arm, next_phase)
        return arm.target_pos, arm.target_quat

    def _do_open_and_wait(self, arm: ArmState, next_phase):
        if arm.phase_step == 1:
            arm.gripper_cmd = self._cfg.gripper.open
        if arm.phase_step > self._cfg.thresh.gripper_wait:
            self._transition(arm, next_phase)
        return arm.target_pos, arm.target_quat

    def _do_lift_from_place(self, arm: ArmState, height: float, next_phase):
        ee = self.get_ee_pose(arm.ee_site)
        if arm.phase_step == 1:
            arm.target_pos = ee[:3] + np.array([0, 0, height])
            arm.target_quat = self._euler_to_quat(*arm.grasp_euler)
        if self._arrived_at_pos(arm, self._cfg.thresh.approach_pos):
            self._transition(arm, next_phase)
        return arm.target_pos, arm.target_quat

    def _do_retreat_synced(
        self,
        arm: ArmState,
        target_pos: np.ndarray,
        check_obj: str | None,
        check_xy: np.ndarray | None,
        next_phase,
        self_phase,
        *,
        check_placement: bool = True,
    ):
        """统一的后撤 + 同步方法。"""
        other = self._other(arm)

        # ── 已到位，等待对方 ──
        if arm.retreat_ready:
            if other.retreat_ready:
                for a in (arm, other):
                    a.phase = next_phase
                    a.phase_step = 0
                    a.retreat_ready = False
                    a.retreat_sub_step = 0
            return arm.target_pos, arm.target_quat

        ee = self.get_ee_pose(arm.ee_site)

        # ── 子步骤 0：先沿 X 方向后撤 ──
        if arm.retreat_sub_step == 0:
            if arm.phase_step == 1:
                arm.target_pos = np.array([target_pos[0], ee[1], ee[2]])
                arm.target_quat = self._euler_to_quat(*arm.grasp_euler)
            if self._arrived_at_pos(arm, self._cfg.thresh.approach_pos):
                arm.retreat_sub_step = 1
                arm.phase_step = 0
            return arm.target_pos, arm.target_quat

        # ── 子步骤 1：沿 Y 方向平移到最终位置 ──
        if arm.phase_step == 1:
            arm.target_pos = target_pos.copy()
            arm.target_quat = self._euler_to_quat(*arm.grasp_euler)

        if not self._arrived_at_pos(arm, self._cfg.thresh.approach_pos):
            return arm.target_pos, arm.target_quat

        if arm.phase_step < self._cfg.thresh.settle_wait:
            return arm.target_pos, arm.target_quat

        if check_placement and check_obj is not None and check_xy is not None:
            obj = self.get_object_pose(check_obj)
            if np.linalg.norm(obj[:2] - check_xy) >= self._cfg.thresh.place_dist:
                return self._retry(arm)

        arm.retreat_ready = True
        arm.phase = self_phase
        return arm.target_pos, arm.target_quat

    # ══════════════════════ 辅助 ══════════════════════

    def _arrived_at_pos(self, arm: ArmState, threshold: float) -> bool:
        ee = self.get_ee_pose(arm.ee_site)
        return bool(np.linalg.norm(arm.target_pos - ee[:3]) < threshold)

    def _object_far_from_ee(self, obj_name: str, ee_pos: np.ndarray) -> bool:
        obj = self.get_object_pose(obj_name)
        return bool(np.linalg.norm(obj[:3] - ee_pos) > self._cfg.thresh.drop_dist)

    def _transition(self, arm: ArmState, next_phase) -> None:
        arm.phase = next_phase
        arm.phase_step = 0

    def _randomize_retreat_pos(self, arm: ArmState) -> None:
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, self._cfg.retreat_radius)
        arm.retreat_pos = np.array([
            arm.retreat_center[0] + radius * np.cos(angle),
            arm.retreat_center[1] + radius * np.sin(angle),
            self._cfg.table_z + self._cfg.heights.retreat,
        ])

    def _retry(self, arm: ArmState):
        arm.retry_count += 1
        if arm.retry_count > self._cfg.thresh.max_retries:
            self.state = TeacherState.FAILURE
            return arm.target_pos, arm.target_quat
        arm.phase = DualPickPhase.MOVE_ABOVE_OWN
        arm.phase_step = 0
        arm.gripper_cmd = self._cfg.gripper.open
        arm.retreat_ready = False
        arm.retreat_sub_step = 0
        arm.center_x = float(np.random.uniform(*self._cfg.center_x_range))
        self._randomize_retreat_pos(arm)
        return arm.target_pos, arm.target_quat

    # ── 评估成功判定（不依赖状态机） ──────────────────

    def check_success(self) -> bool:
        """成功 = 双臂各自把物块放回自己的盘子上：
        A_cube 在 A_plate 上，且 B_cube 在 B_plate 上。
        """
        a = self._arm_a
        b = self._arm_b
        ok_a = self._cube_on_plate(a.own_cube, a.own_plate)
        ok_b = self._cube_on_plate(b.own_cube, b.own_plate)
        return bool(ok_a and ok_b)

    def _cube_on_plate(self, cube_name: str, plate_name: str) -> bool:
        cube = self.get_object_pose(cube_name)
        plate = self.get_object_pose(plate_name)
        return bool(np.linalg.norm(cube[:2] - plate[:2]) < self._cfg.thresh.place_dist)

    # ══════════════════════ 姿态计算 ══════════════════════

    @staticmethod
    def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ])

    def _compute_grasp_yaw(self, arm: ArmState, cube_quat: np.ndarray) -> float:
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
        ee = self.get_ee_pose(arm.ee_site)
        cy = self._quat_to_yaw(ee[3:])
        return min(candidates, key=lambda t: min(
            abs(t - cy), abs(t - cy + 2 * np.pi), abs(t - cy - 2 * np.pi),
        ))

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

    def _make_grasp_quat(self, extra_yaw: float) -> np.ndarray:
        base = self._euler_to_quat(np.pi, 0.0, 0.0)
        cy, sy = np.cos(extra_yaw / 2), np.sin(extra_yaw / 2)
        yaw_q = np.array([cy, 0, 0, sy])
        return np.array([
            yaw_q[0] * base[0] - yaw_q[1] * base[1] - yaw_q[2] * base[2] - yaw_q[3] * base[3],
            yaw_q[0] * base[1] + yaw_q[1] * base[0] + yaw_q[2] * base[3] - yaw_q[3] * base[2],
            yaw_q[0] * base[2] - yaw_q[1] * base[3] + yaw_q[2] * base[0] + yaw_q[3] * base[1],
            yaw_q[0] * base[3] + yaw_q[1] * base[2] - yaw_q[2] * base[1] + yaw_q[3] * base[0],
        ])
