"""仿真控制器。

Controller 协议定义统一控制接口，所有控制器返回 joint-level 动作。
内置实现：
  - ScriptedTeacherController — 封装 scripted teacher + MinkIK 流水线
  - KeyboardTeleopController  — 键盘遥操作（增量目标位姿 + IK）
"""

from __future__ import annotations

import threading
from typing import Any, Protocol, runtime_checkable

import mujoco
import numpy as np

from ..configs.config_loader import SceneConfig
from ..simulate.ik_solver import MinkIK
from .teachers import TEACHER_REGISTRY, create_teacher


@runtime_checkable
class Controller(Protocol):
    def reset(self) -> None: ...
    def step(self, observation: dict[str, Any]) -> np.ndarray: ...
    def is_done(self) -> bool: ...
    def is_success(self) -> bool: ...


class ScriptedTeacherController:
    """封装 scripted teacher + MinkIK 流水线。

    每个 arm 独立维护一个 teacher + IK 求解器。
    teacher 输出目标位姿 → IK 求解 → 返回 joint-level 动作。
    """

    def __init__(
        self,
        config: SceneConfig,
        teacher_config: Any,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        tcls = TEACHER_REGISTRY.get(teacher_config.type)
        if tcls is None:
            raise ValueError(f"未知 teacher 类型: {teacher_config.type!r}")
        self._robots = list(config.robots)
        self._policy_dt = config.sim.policy_dt
        self._multi_arm = bool(getattr(tcls, "_is_multi_arm", False))

        self._iks: dict[str, MinkIK] = {}
        for r in self._robots:
            self._iks[r.prefix] = MinkIK(
                model,
                init_qpos=data.qpos.copy(),
                dt=self._policy_dt,
                ee_site_name=r.prefixed_ee_site,
                vel_limit=[10.0] * 6,
                arm_joint_names=r.prefixed_arm_joints,
            )

        if self._multi_arm:
            self._teacher: Any = create_teacher(
                teacher_config.type, model, data, config=teacher_config
            )
        else:
            self._teachers: dict[str, Any] = {}
            for r in self._robots:
                self._teachers[r.prefix] = create_teacher(
                    teacher_config.type, model, data,
                    config=teacher_config, prefix=r.prefix,
                )

    # ── Controller 接口 ────────────────────────────────

    def reset(self) -> None:
        if self._multi_arm:
            self._teacher.reset()
        else:
            for t in self._teachers.values():
                t.reset()

    def step(self, observation: dict[str, Any]) -> np.ndarray:
        arm_joint_pos = np.asarray(observation["arm_joint_pos"], dtype=np.float64)

        if self._multi_arm:
            tgt_dict = self._teacher.step()
        else:
            tgt_dict: dict[str, np.ndarray] = {}
            for prefix, t in self._teachers.items():
                tgt_dict.update(t.step())

        actions: list[np.ndarray] = []
        offset = 0
        for r in self._robots:
            tgt = np.asarray(tgt_dict[r.prefix], dtype=np.float64)
            target_pose = tgt[:7]
            grip_cmd = float(tgt[7])

            n_arm = r.n_arm_joints
            cur_arm_qpos = arm_joint_pos[offset : offset + n_arm]
            jt = self._iks[r.prefix].solve(
                cur_arm_qpos, target_pose, dt=self._policy_dt
            )
            actions.append(jt.astype(np.float32))
            actions.append(np.array([grip_cmd], dtype=np.float32))
            offset += n_arm

        return np.concatenate(actions).astype(np.float32)

    def is_done(self) -> bool:
        if self._multi_arm:
            return self._teacher.is_done()
        return all(t.is_done() for t in self._teachers.values())

    def is_success(self) -> bool:
        if self._multi_arm:
            return self._teacher.is_success()
        return all(t.is_success() for t in self._teachers.values())

    def check_success(self) -> bool:
        """评估成功判定：基于物理状态，由各 teacher 的 check_success() 汇总。"""
        if self._multi_arm:
            return bool(self._teacher.check_success())
        return all(t.check_success() for t in self._teachers.values())


class KeyboardTeleopController:
    """键盘遥操作控制器 — 增量目标位姿 + IK。

    按键（由 mujoco viewer 的 key_callback 线程调用 key_event）：
      W/S       +x / -x
      A/D       +y / -y
      R/F       +z / -z
      上/下      +roll / -roll
      左/右      +yaw / -yaw
      Q/E       +pitch / -pitch
      Space     切换夹爪开合
    """

    def __init__(
        self,
        config: SceneConfig,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        pos_step: float = 0.01,
        rot_step: float = 0.1,
    ) -> None:
        self._robots = list(config.robots)
        self._policy_dt = config.sim.policy_dt
        self._pos_step = pos_step
        self._rot_step = rot_step

        self._iks: dict[str, MinkIK] = {}
        self._target_pos: dict[str, np.ndarray] = {}
        self._target_quat: dict[str, np.ndarray] = {}
        self._gripper: dict[str, float] = {}
        self._lock = threading.Lock()
        self._model = model
        self._data = data

        for r in self._robots:
            self._iks[r.prefix] = MinkIK(
                model,
                init_qpos=data.qpos.copy(),
                dt=self._policy_dt,
                ee_site_name=r.prefixed_ee_site,
                vel_limit=[10.0] * 6,
                arm_joint_names=r.prefixed_arm_joints,
            )
            ee = self._ee_pose(r.prefixed_ee_site)
            self._target_pos[r.prefix] = ee[:3].copy()
            self._target_quat[r.prefix] = ee[3:7].copy()
            self._gripper[r.prefix] = 0.0

    def _ee_pose(self, site_name: str) -> np.ndarray:
        ee = np.zeros(7)
        try:
            site_id = self._model.site(site_name).id
            pos = self._data.site_xpos[site_id]
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, self._data.site_xmat[site_id])
            ee = np.concatenate([pos, quat])
        except Exception:
            pass
        return ee

    # ── Controller 接口 ────────────────────────────────

    def reset(self) -> None:
        with self._lock:
            for r in self._robots:
                ee = self._ee_pose(r.prefixed_ee_site)
                self._target_pos[r.prefix] = ee[:3].copy()
                self._target_quat[r.prefix] = ee[3:7].copy()
                self._gripper[r.prefix] = 0.0

    def key_event(self, key: int) -> bool:
        """处理一个按键事件，返回是否消费（产生动作）。"""
        with self._lock:
            for r in self._robots:
                prefix = r.prefix
                pos = self._target_pos[prefix]
                quat = self._target_quat[prefix]
                if key == 87:   # W
                    pos[0] += self._pos_step
                elif key == 83:  # S
                    pos[0] -= self._pos_step
                elif key == 68:  # D
                    pos[1] += self._pos_step
                elif key == 65:  # A
                    pos[1] -= self._pos_step
                elif key == 82:  # R
                    pos[2] += self._pos_step
                elif key == 70:  # F
                    pos[2] -= self._pos_step
                elif key == 81:  # Q
                    quat[:] = self._apply_axis_rot(quat, [0, 0, 1], self._rot_step)
                elif key == 69:  # E
                    quat[:] = self._apply_axis_rot(quat, [0, 0, 1], -self._rot_step)
                elif key == 265:  # Up
                    quat[:] = self._apply_axis_rot(quat, [1, 0, 0], self._rot_step)
                elif key == 264:  # Down
                    quat[:] = self._apply_axis_rot(quat, [1, 0, 0], -self._rot_step)
                elif key == 263:  # Left
                    quat[:] = self._apply_axis_rot(quat, [0, 1, 0], self._rot_step)
                elif key == 262:  # Right
                    quat[:] = self._apply_axis_rot(quat, [0, 1, 0], -self._rot_step)
                elif key == 32:   # Space
                    self._gripper[prefix] = 0.8 if self._gripper[prefix] < 0.4 else 0.0
                else:
                    return False
        return True

    @staticmethod
    def _apply_axis_rot(quat: np.ndarray, axis: list[float], angle: float) -> np.ndarray:
        import math
        ha = angle / 2
        dq = np.array([
            math.cos(ha),
            axis[0] * math.sin(ha),
            axis[1] * math.sin(ha),
            axis[2] * math.sin(ha),
        ])
        # 四元数乘法 q' = dq ⊗ q
        q0, q1, q2, q3 = dq
        r0, r1, r2, r3 = quat
        return np.array([
            q0 * r0 - q1 * r1 - q2 * r2 - q3 * r3,
            q0 * r1 + q1 * r0 + q2 * r3 - q3 * r2,
            q0 * r2 - q1 * r3 + q2 * r0 + q3 * r1,
            q0 * r3 + q1 * r2 - q2 * r1 + q3 * r0,
        ])

    def step(self, observation: dict[str, Any]) -> np.ndarray:
        arm_joint_pos = np.asarray(observation["arm_joint_pos"], dtype=np.float64)
        actions: list[np.ndarray] = []
        offset = 0
        with self._lock:
            for r in self._robots:
                n_arm = r.n_arm_joints
                cur = arm_joint_pos[offset : offset + n_arm]
                target_pose = np.concatenate(
                    [self._target_pos[r.prefix], self._target_quat[r.prefix]]
                )
                jt = self._iks[r.prefix].solve(cur, target_pose, dt=self._policy_dt)
                actions.append(jt.astype(np.float32))
                actions.append(np.array([self._gripper[r.prefix]], dtype=np.float32))
                offset += n_arm
        return np.concatenate(actions).astype(np.float32)

    def is_done(self) -> bool:
        return False

    def is_success(self) -> bool:
        return False
