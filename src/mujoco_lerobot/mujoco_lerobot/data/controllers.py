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
from .recording import RecordingDecision
from .teachers import TEACHER_REGISTRY, create_teacher


@runtime_checkable
class Controller(Protocol):
    def reset(self) -> None: ...
    def step(self, observation: dict[str, Any]) -> np.ndarray: ...
    def is_done(self) -> bool: ...
    def is_success(self) -> bool: ...

    def recording_decision(self, recording: bool) -> RecordingDecision | None:
        """每策略步由 run_episode 调用，控制录制生命周期。

        recording=False 表示尚未开始录制（teacher 可返回 START 请求开始，
        或返回 QUIT 直接退出）；recording=True 时返回 SAVED / DISCARDED /
        QUIT 结束本集，返回 None 继续。
        """
        ...


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
        teacher_kwargs: dict[str, Any] | None = None,
    ) -> None:
        tcls = TEACHER_REGISTRY.get(teacher_config.type)
        if tcls is None:
            raise ValueError(f"未知 teacher 类型: {teacher_config.type!r}")
        self._robots = list(config.robots)
        self._policy_dt = config.sim.policy_dt
        self._multi_arm = bool(getattr(tcls, "_is_multi_arm", False))
        teacher_kwargs = teacher_kwargs or {}

        self._iks: dict[str, MinkIK] = {}
        for r in self._robots:
            self._iks[r.prefix] = MinkIK(
                model,
                init_qpos=data.qpos.copy(),
                dt=self._policy_dt,
                ee_site_name=r.prefixed_ee_site,
                arm_joint_names=r.prefixed_arm_joints,
                ik_config=r.ik_solver,
                prefix=r.prefix,
            )

        if self._multi_arm:
            self._teacher: Any = create_teacher(
                teacher_config.type, model, data,
                config=teacher_config, **teacher_kwargs,
            )
        else:
            self._teachers: dict[str, Any] = {}
            for r in self._robots:
                self._teachers[r.prefix] = create_teacher(
                    teacher_config.type, model, data,
                    config=teacher_config, prefix=r.prefix, **teacher_kwargs,
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

    def recording_decision(self, recording: bool) -> RecordingDecision | None:
        """聚合各臂 teacher 的录制决策（优先级 QUIT > DISCARDED > SAVED > START）。"""
        if self._multi_arm:
            return self._teacher.recording_decision(recording)
        decisions = [t.recording_decision(recording) for t in self._teachers.values()]
        non_null = [d for d in decisions if d is not None]
        if not non_null:
            return None
        for d in (
            RecordingDecision.QUIT,
            RecordingDecision.DISCARDED,
            RecordingDecision.SAVED,
            RecordingDecision.START,
        ):
            if d in non_null:
                return d
        return None

    # ── 采集会话钩子（透传给各臂 teacher） ──────────────

    @property
    def retry_limit(self) -> int | None:
        """每 episode 尝试上限；全为 None 时由采集脚本用场景配置。"""
        if self._multi_arm:
            return self._teacher.retry_limit
        limits = [t.retry_limit for t in self._teachers.values()]
        non_null = [l for l in limits if l is not None]
        return min(non_null) if non_null else None

    def start_collection(self) -> None:
        if self._multi_arm:
            self._teacher.start_collection()
        else:
            for t in self._teachers.values():
                t.start_collection()

    def end_collection(self) -> None:
        if self._multi_arm:
            self._teacher.end_collection()
        else:
            for t in self._teachers.values():
                t.end_collection()


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

        # 录制控制事件（Enter=保存 Backspace=丢弃 Esc=退出），由按键线程写入
        self._rec_save = False
        self._rec_discard = False
        self._rec_quit = False

        for r in self._robots:
            self._iks[r.prefix] = MinkIK(
                model,
                init_qpos=data.qpos.copy(),
                dt=self._policy_dt,
                ee_site_name=r.prefixed_ee_site,
                arm_joint_names=r.prefixed_arm_joints,
                ik_config=r.ik_solver,
                prefix=r.prefix,
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

    # ── 录制控制（由 run_episode 每策略步询问） ─────────

    def record_event(self, key: int) -> bool:
        """处理录制控制键（Enter=保存 Backspace=丢弃 Esc=退出），返回是否消费。

        由 viewer 按键线程调用；与移动键（key_event）互斥锁共享。
        """
        with self._lock:
            if key == 256:  # Esc
                self._rec_quit = True
            elif key == 13:  # Enter
                self._rec_save = True
            elif key in (8, 127):  # Backspace
                self._rec_discard = True
            else:
                return False
        return True

    def recording_decision(self, recording: bool) -> RecordingDecision | None:
        """键盘遥操作：自动开始录制；Enter=保存 / Backspace=丢弃 / Esc=退出。"""
        with self._lock:
            if self._rec_quit:
                return RecordingDecision.QUIT
            if not recording:
                return RecordingDecision.START  # 键盘模式 reset 后立即开始录制
            if self._rec_save:
                self._rec_save = False
                return RecordingDecision.SAVED
            if self._rec_discard:
                self._rec_discard = False
                return RecordingDecision.DISCARDED
        return None
