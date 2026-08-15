"""逆运动学求解器（基于 mink 库）。

将绝对目标位姿 [x,y,z, qw,qx,qy,qz] 转换为关节位置目标。
使用 mink 的 QP 求解器进行 differential IK。
"""

from __future__ import annotations

import mujoco
import numpy as np
import mink

from ..configs.config_loader import IKSolverConfig

DEFAULT_ARM_JOINTS = [
    "ur_shoulder_pan_joint",
    "ur_shoulder_lift_joint",
    "ur_elbow_joint",
    "ur_wrist_1_joint",
    "ur_wrist_2_joint",
    "ur_wrist_3_joint",
]

DEFAULT_VEL_LIMIT = [3.1416] * 6


class MinkIK:
    """基于 mink 的逆运动学求解器。

    每次 solve() 调用 mink.solve_ik() 求解 QP，
    然后用 config.integrate_inplace() 更新内部状态，
    返回 arm 关节目标位置。

    支持两种参数方式：
      - 散装参数（兼容旧调用，如 tests/test_simulate.py）
      - ``ik_config: IKSolverConfig``（优先，来自场景 yaml ``robot/ik_solver``），
        提供时覆盖全部散装参数（含 vel_limit），并启用碰撞避免等扩展。
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
        ik_config: IKSolverConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.model = model
        self._prefix = prefix

        # ── 参数解析：ik_config 优先，散装参数为回退 ──
        if ik_config is not None:
            vel_limit = ik_config.vel_limit
            pos_cost = ik_config.pos_cost
            ori_cost = ik_config.ori_cost
            posture_cost = ik_config.posture_cost
            lm_damping = ik_config.lm_damping
            solver = ik_config.solver
            self._gain = ik_config.gain
            self._damping = ik_config.damping
            self._safety_break = ik_config.safety_break
            self._max_iters = max(1, int(ik_config.max_iters))
            self._pos_threshold = ik_config.pos_threshold
            self._ori_threshold = ik_config.ori_threshold
            self._collision_avoidance = ik_config.collision_avoidance
        else:
            self._gain = 1.0
            self._damping = 1e-12
            self._safety_break = False
            self._max_iters = 1
            self._pos_threshold = 0.01
            self._ori_threshold = 0.1
            self._collision_avoidance = None

        self._solver = solver
        self._dt = dt
        self._vel_limit = (
            vel_limit if vel_limit is not None else list(DEFAULT_VEL_LIMIT)
        )

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
            gain=self._gain,
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
        self._add_collision_avoidance()

    # ── 碰撞避免 ────────────────────────────────────────

    def _add_collision_avoidance(self) -> None:
        """按 ik_config.collision_avoidance 配置添加 CollisionAvoidanceLimit。"""
        if self._collision_avoidance is None or not self._collision_avoidance.enabled:
            return
        ca = self._collision_avoidance
        if not ca.pairs:
            raise ValueError(
                "collision_avoidance.enabled=true 但未配置 pairs（示例："
                "[['ur_wrist_3_link'], ['table_surface']]）"
            )
        geom_pairs = [
            (
                self._resolve_geom_ids(self.model, self._prefix, list(a)),
                self._resolve_geom_ids(self.model, self._prefix, list(b)),
            )
            for a, b in ca.pairs
        ]
        self._limits.append(
            mink.CollisionAvoidanceLimit(
                model=self.model,
                geom_pairs=geom_pairs,
                gain=ca.gain,
                minimum_distance_from_collisions=ca.minimum_distance,
                collision_detection_distance=ca.detection_distance,
                bound_relaxation=ca.bound_relaxation,
                broadphase=ca.broadphase,
            )
        )

    @staticmethod
    def _resolve_geom_ids(
        model: mujoco.MjModel, prefix: str, tokens: list[str]
    ) -> list[int]:
        """把配置中的 geom 名解析为模型 geom id 列表。

        匹配优先级（对每个 token）：
          1. 精确名（如 ``A_COLLISION_ur_wrist_3_link_0``）
          2. ``{prefix}COLLISION_{token}`` 前缀（简短名如 ``ur_wrist_3_link``）
          3. ``{prefix}{token}`` 前缀
          4. ``{token}`` 前缀（对侧臂显式名如 ``B_ur_wrist_3_link``）
        无匹配抛 ValueError，提示可用 geom。
        """
        all_names = [model.geom(i).name for i in range(model.ngeom)]
        name_set = set(all_names)
        ids: list[int] = []
        for token in tokens:
            matched: list[str] = []
            if token in name_set:
                matched = [token]
            else:
                for p in (f"{prefix}COLLISION_{token}", f"{prefix}{token}", token):
                    matched = [n for n in all_names if n.startswith(p)]
                    if matched:
                        break
            if not matched:
                raise ValueError(
                    f"碰撞避免 geom {token!r} 未匹配到模型中的任何 geom "
                    f"(prefix={prefix!r})。模型 geom 示例: {sorted(name_set)[:16]}"
                )
            ids.extend(model.geom(n).id for n in sorted(matched))
        # 去重保序
        return list(dict.fromkeys(ids))

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

        # QP 求解 → 积分（多迭代收敛，max_iters=1 等价单次求解）
        for _ in range(self._max_iters):
            vel = mink.solve_ik(
                self._config,
                [self._ee_task, self._posture_task],
                dt=float(dt),
                solver=self._solver,
                damping=self._damping,
                safety_break=self._safety_break,
                limits=self._limits,
            )
            self._config.integrate_inplace(vel, float(dt))
            if self._max_iters > 1:
                err = self._ee_task.compute_error(self._config)
                pos_ok = np.linalg.norm(err[:3]) <= self._pos_threshold
                ori_ok = np.linalg.norm(err[3:]) <= self._ori_threshold
                if pos_ok and ori_ok:
                    break

        # 返回 arm 关节位置
        return self._config.q[self._arm_qpos_adr].copy()
