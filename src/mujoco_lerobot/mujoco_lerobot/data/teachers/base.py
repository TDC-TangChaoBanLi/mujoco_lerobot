"""Scripted Teacher 基类。

所有 scripted teacher 继承此类，提供状态机框架和工具方法。
step() 返回 dict[str, np.ndarray]: {arm_prefix: [x,y,z,qw,qx,qy,qz,gripper_cmd], ...}
由 IK 求解器转换为关节级控制命令。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import mujoco
import numpy as np


class TeacherState(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class Teacher:
    """Scripted teacher 基类。

    子类需要实现 step() → dict[str, np.ndarray]，每个键为一个臂前缀，
    值为 [x,y,z,qw,qx,qy,qz,gripper_cmd]。

    注册机制：
      - 子类通过 `teacher_type`（配置 yaml 中 `type:` 的值）在 TEACHER_REGISTRY
        中注册；`@register_teacher("XxxTeacher")` 装饰器自动完成注册。
      - `config_class` 指定对应的 teacher 配置 dataclass，供配置加载与实例化。
      - 外部新增任务时，只需在对应模块里实现子类并用装饰器注册即可自动发现。
    """

    _is_multi_arm: bool = False  # True 表示单实例控制多臂

    # 注册元数据（子类覆盖）
    teacher_type: str = ""
    config_class: Any = None

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: Any | None = None,
        prefix: str = "",
    ) -> None:
        self.model = model
        self.data = data
        self.config = config
        self.prefix = prefix
        self.state = TeacherState.RUNNING
        self.current_step = 0

    def reset(self) -> None:
        self.state = TeacherState.RUNNING
        self.current_step = 0

    def step(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def is_success(self) -> bool:
        return self.state == TeacherState.SUCCESS

    def is_failure(self) -> bool:
        return self.state == TeacherState.FAILURE

    def is_done(self) -> bool:
        return self.state != TeacherState.RUNNING

    # ── 评估成功判定（不依赖状态机） ────────────────────

    def check_success(self) -> bool:
        """基于当前物理状态判断任务是否已完成，供评估环境调用。

        与 is_success()（teacher 状态机是否跑完）不同，本方法应直接
        根据 model/data 中物体位姿等物理状态判定任务达成情况，
        使得任何策略（而不只是本 teacher）在评估时都能用同一标准判断成功。
        子类必须实现。
        """
        raise NotImplementedError

    # ── 位姿查询 ───────────────────────────────────────

    def get_ee_pose(self, site_name: str = "_tcp") -> np.ndarray:
        """获取指定 site 的末端位姿 [x,y,z, qw,qx,qy,qz]。"""
        try:
            site_id = self.model.site(site_name).id
            pos = self.data.site_xpos[site_id]
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id])
            return np.concatenate([pos, quat])
        except Exception:
            return np.zeros(7)

    def get_object_pose(self, name: str) -> np.ndarray:
        """获取物体位姿 [x,y,z, qw,qx,qy,qz]。"""
        try:
            body_id = self.model.body(name).id
            pos = self.data.xpos[body_id]
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, self.data.xmat[body_id])
            return np.concatenate([pos, quat])
        except Exception:
            return np.zeros(7)

    # ── 增量计算 ───────────────────────────────────────

    def compute_delta_pos(
        self, target: np.ndarray, current: np.ndarray, speed: float = 0.01,
    ) -> np.ndarray:
        delta = target - current
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return np.zeros(3)
        return np.clip(delta / dist * speed, -speed, speed)

    def compute_delta_rot(
        self,
        target_quat: np.ndarray,
        current_quat: np.ndarray,
        speed: float = 0.05,
    ) -> np.ndarray:
        ori_err = np.zeros(3)
        mujoco.mju_subQuat(ori_err, target_quat, current_quat)
        angle = np.linalg.norm(ori_err)
        if angle < 1e-6:
            return np.zeros(3)
        return np.clip(ori_err / angle * min(angle, speed), -speed, speed)

    # ── 动作构建 ───────────────────────────────────────

    def make_action(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        gripper_cmd: float = 0.0,
    ) -> np.ndarray:
        """构建绝对位姿动作 [x,y,z, qw,qx,qy,qz, gripper]。"""
        action = np.zeros(8, dtype=np.float64)
        action[:3] = target_pos
        if target_quat is not None:
            action[3:7] = target_quat
        action[7] = gripper_cmd
        return action

    @staticmethod
    def euler_to_quat(
        roll: float, pitch: float, yaw: float,
    ) -> np.ndarray:
        quat = np.empty(4)
        mujoco.mju_euler2Quat(quat, [roll, pitch, yaw], "XYZ")
        return quat
