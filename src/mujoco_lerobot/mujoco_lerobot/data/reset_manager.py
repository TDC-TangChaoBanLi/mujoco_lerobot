"""多臂多物体环境重置管理器。

负责：
- 将各臂重置到 default_qpos（qpos + actuator ctrl 同步）
- 随机化物体位姿（x/y/z 范围 + 欧拉角范围）
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..configs.config_loader import ObjectRandomization, RobotConfig
from ..simulate.mujoco_wrapper import MujocoWrapper


class ResetManager:
    def __init__(
        self,
        mj: MujocoWrapper,
        robots: list[RobotConfig],
        objects: dict[str, ObjectRandomization],
    ) -> None:
        self.mj = mj
        self.robots = robots
        self.objects = objects

        # 预计算各臂 actuator ID
        self._actuator_ids: dict[str, tuple[list[int], list[int]]] = {}
        for r in robots:
            arm_ids = [
                mj.get_actuator_id(f"{j}_ACTUATOR") for j in r.prefixed_arm_joints
            ]
            grip_ids = [
                mj.get_actuator_id(f"{j}_ACTUATOR") for j in r.prefixed_gripper_joints
            ]
            self._actuator_ids[r.prefix] = (arm_ids, grip_ids)

        # 预计算物体 freejoint 的 qposadr
        self._object_qposadr: dict[str, int] = {}
        for name in objects:
            jid = mj.get_body_joint_id(name)
            if jid is not None:
                self._object_qposadr[name] = int(mj.model.jnt_qposadr[jid])

    def reset(self, *, randomize_objects: bool = True) -> None:
        """完整重置流程：mj_resetData → 臂到位 → 随机化物体 → 同步 ctrl → forward。"""
        mujoco.mj_resetData(self.mj.model, self.mj.data)

        ctrl = self.mj.get_ctrl()
        for r in self.robots:
            arm_ids, grip_ids = self._actuator_ids[r.prefix]
            # 设 actuator 目标使 arm 初始到位
            ctrl[arm_ids] = np.asarray(r.default_qpos, dtype=np.float64)
            ctrl[grip_ids] = 0.0
            # 设 qpos 使 arm 立即到位
            for i, jname in enumerate(r.prefixed_arm_joints):
                self.mj.set_joint_qpos(jname, r.default_qpos[i])
        self.mj.set_ctrl(ctrl)

        if randomize_objects:
            self._randomize_objects()

        mujoco.mj_forward(self.mj.model, self.mj.data)

    def _randomize_objects(self) -> None:
        for obj_name, rand in self.objects.items():
            adr = self._object_qposadr.get(obj_name)
            if adr is None:
                continue
            x = np.random.uniform(*rand.x_range)
            y = np.random.uniform(*rand.y_range)
            z = np.random.uniform(*rand.z_range)
            self.mj.data.qpos[adr : adr + 3] = [x, y, z]

            has_rot = (
                rand.roll_range != (0.0, 0.0)
                or rand.pitch_range != (0.0, 0.0)
                or rand.yaw_range != (0.0, 0.0)
            )
            if has_rot:
                roll = np.random.uniform(*rand.roll_range)
                pitch = np.random.uniform(*rand.pitch_range)
                yaw = np.random.uniform(*rand.yaw_range)
                quat = np.empty(4)
                mujoco.mju_euler2Quat(quat, [roll, pitch, yaw], "XYZ")
                self.mj.data.qpos[adr + 3 : adr + 7] = quat  # [qw, qx, qy, qz]
