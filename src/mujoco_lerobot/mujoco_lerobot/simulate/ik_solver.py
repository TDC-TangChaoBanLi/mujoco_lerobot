"""逆运动学求解器（基于 mink 库）。

将绝对目标位姿 [x,y,z, qw,qx,qy,qz] 转换为关节位置目标。
使用 mink 的 QP 求解器进行 differential IK。
"""

from __future__ import annotations

import mujoco
import numpy as np
import mink

DEFAULT_ARM_JOINTS = [
    "ur_shoulder_pan_joint",
    "ur_shoulder_lift_joint",
    "ur_elbow_joint",
    "ur_wrist_1_joint",
    "ur_wrist_2_joint",
    "ur_wrist_3_joint",
]


class MinkIK:
    """基于 mink 的逆运动学求解器。

    每次 solve() 调用 mink.solve_ik() 求解 QP，
    然后用 config.integrate_inplace() 更新内部状态，
    返回 arm 关节目标位置。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        init_qpos: np.ndarray,
        dt: float = 0.01,
        ee_site_name: str = "_tcp",
        pos_cost: float = 1.0,
        ori_cost: float = 1.0,
        vel_limit: list[float] | None = None,
        posture_cost: float = 1e-3,
        lm_damping: float = 1e-6,
        arm_joint_names: list[str] | None = None,
        solver: str = "daqp",
    ) -> None:
        self.model = model
        self._solver = solver
        self._dt = dt
        self._vel_limit = vel_limit if vel_limit is not None else [10.0] * 6

        if arm_joint_names is None:
            arm_joint_names = list(DEFAULT_ARM_JOINTS)
        self.arm_joint_names = arm_joint_names
        self._arm_qpos_adr = [
            model.jnt_qposadr[model.joint(n).id] for n in arm_joint_names
        ]

        # mink Configuration（内部持有独立的 MjData）
        self._config = mink.Configuration(model)
        self._config.update(np.asarray(init_qpos, dtype=np.float64))

        # 任务
        self._ee_task = mink.FrameTask(
            frame_name=ee_site_name,
            frame_type="site",
            position_cost=pos_cost,
            orientation_cost=ori_cost,
            lm_damping=lm_damping,
        )
        self._posture_task = mink.PostureTask(model, cost=posture_cost)
        self._posture_task.set_target(self._config.q)

        # 约束
        self._limits = [
            mink.ConfigurationLimit(model=model),
            mink.VelocityLimit(
                model=model,
                velocities={
                    name: vlimit
                    for name, vlimit in zip(self.arm_joint_names, self._vel_limit)
                },
            ),
        ]

    def reset(self, qpos: np.ndarray) -> None:
        """重置内部状态到指定关节角。"""
        self._config.update(np.asarray(qpos, dtype=np.float64))
        self._posture_task.set_target(self._config.q)

    def solve(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        dt: float = 0.01,
    ) -> np.ndarray:
        """求解 IK 返回 arm 关节目标位置 (n_arm,)。

        Args:
            current_qpos: 当前关节位置，如只含 arm 关节会自动扩展为完整 qpos。
            target_pose: [x, y, z, qw, qx, qy, qz]
            dt: 时间步长

        Returns:
            arm 关节目标位置 np.float64
        """
        # 构建完整 qpos
        if current_qpos.shape[0] == len(self._arm_qpos_adr):
            qpos = np.zeros(self.model.nq)
            for i, adr in enumerate(self._arm_qpos_adr):
                qpos[adr] = current_qpos[i]
        else:
            qpos = np.asarray(current_qpos, dtype=np.float64).copy()

        # 同步 mink 内部状态
        self._config.update(qpos)
        self._posture_task.set_target(self._config.q)

        # 设置目标
        tgt_quat = np.asarray(target_pose[3:7], dtype=np.float64)
        tgt_pos = np.asarray(target_pose[:3], dtype=np.float64)
        self._ee_task.set_target(
            mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(wxyz=tgt_quat),
                translation=tgt_pos,
            )
        )

        # QP 求解 → 速度
        vel = mink.solve_ik(
            self._config,
            [self._ee_task, self._posture_task],
            dt=float(dt),
            solver=self._solver,
            limits=self._limits,
        )

        # 积分
        self._config.integrate_inplace(vel, float(dt))

        # 返回 arm 关节位置
        return self._config.q[self._arm_qpos_adr].copy()
