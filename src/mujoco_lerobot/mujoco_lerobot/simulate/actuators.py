"""共享执行器辅助 — 计算 actuator 映射并应用关节动作。

供数据采集（SimulationManager）与评估环境（gym env）复用，
避免重复实现「flat action → ctrl」的映射逻辑。
"""

from __future__ import annotations

import numpy as np

from ..configs.config_loader import RobotConfig
from .mujoco_wrapper import MujocoWrapper


def build_actuator_mapping(
    mj: MujocoWrapper,
    robots: list[RobotConfig],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """返回 (arm_act_ids, grip_act_ids)：每臂前缀 → actuator id 列表。"""
    arm_act_ids: dict[str, list[int]] = {}
    grip_act_ids: dict[str, list[int]] = {}
    for r in robots:
        arm_act_ids[r.prefix] = [
            mj.get_actuator_id(f"{j}_ACTUATOR") for j in r.prefixed_arm_joints
        ]
        grip_act_ids[r.prefix] = [
            mj.get_actuator_id(f"{j}_ACTUATOR") for j in r.prefixed_gripper_joints
        ]
    return arm_act_ids, grip_act_ids


def apply_arm_action(
    mj: MujocoWrapper,
    robots: list[RobotConfig],
    arm_act_ids: dict[str, list[int]],
    grip_act_ids: dict[str, list[int]],
    action: np.ndarray,
) -> None:
    """将 flat action 写入 MuJoCo ctrl。

    action layout: [arm0_joints..., grip0..., arm1_joints..., grip1..., ...]
    """
    ctrl = mj.get_ctrl()
    arr = np.asarray(action, dtype=np.float64).ravel()
    offset = 0
    for r in robots:
        arm_ids = arm_act_ids[r.prefix]
        grip_ids = grip_act_ids[r.prefix]
        n_arm = len(arm_ids)
        for j, aid in enumerate(arm_ids):
            if offset + j < len(arr):
                ctrl[aid] = arr[offset + j]
        offset += n_arm
        for j, gid in enumerate(grip_ids):
            if offset + j < len(arr):
                ctrl[gid] = arr[offset + j]
        offset += len(grip_ids)
    mj.set_ctrl(ctrl)
