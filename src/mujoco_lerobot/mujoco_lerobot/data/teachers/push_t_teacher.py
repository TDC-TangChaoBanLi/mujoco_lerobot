"""PushT 鼠标遥操作 Teacher — 人类通过 2D 俯视窗口控制 TCP 推动 T 物块。

该 teacher 不是状态机自动脚本，而是**人类遥操作的桥接器**：
- step() 输出期望 TCP 的绝对位姿（读自 TeleopState.desired，来自鼠标/滚轮），
  姿态固定为夹爪朝下，经 ScriptedTeacherController 的 IK 转为关节动作。
- 自身只依赖 TeleopState（线程安全的纯逻辑，不依赖 pygame），可无头测试；
  TeleopWindow（pygame）由采集脚本创建并共享同一个 TeleopState。
- publish_teleop_state() 由仿真线程每策略步调用，把物理状态（真实 TCP、
  T 物块位姿/yaw、目标位姿、可推动性）回写窗口用于渲染与状态栏。
- check_success() 按 T 中心到目标中心的距离 + yaw 差判定成功，
  供评估环境（lerobot-eval）使用。
"""

from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

from ...configs.teacher_config import PushTTeacherConfig
from ..recording import RecordingDecision
from .push_t_teleop import TeleopState, TeleopParams
from . import register_teacher
from .base import Teacher


def _quat_to_yaw(quat: np.ndarray) -> float:
    """绕世界 z 轴的偏航角（T 物块朝向）。"""
    qw, qx, qy, qz = quat
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


@register_teacher("PushTTeacher")
class PushTTeacher(Teacher):
    teacher_type = "PushTTeacher"
    config_class = PushTTeacherConfig

    # 遥操作：discarded 是用户主动丢弃，采集脚本不重试，直接进入下一集
    retry_limit = 1

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: PushTTeacherConfig | None = None,
        prefix: str = "",
        state: TeleopState | None = None,
    ) -> None:
        super().__init__(model, data, config=config, prefix=prefix)
        cfg = config or PushTTeacherConfig()
        self._cfg = cfg
        self._grasp_quat = np.asarray(cfg.grasp_quat, dtype=np.float64)
        self._state: TeleopState = state or TeleopState(
            TeleopParams(
                window_width=cfg.window_width,
                window_height=cfg.window_height,
                workspace_x=cfg.workspace_x,
                workspace_y=cfg.workspace_y,
                mouse_sens=cfg.sens_mouse,
                wheel_sens=cfg.sens_wheel,
                sens_step=cfg.sens_step,
                invert_mouse_x=cfg.sens_invert_x,
                invert_mouse_y=cfg.sens_invert_y,
                tcp_z_min=cfg.tcp_z_min,
                tcp_z_max=cfg.tcp_z_max,
                push_z_min=cfg.push_z_min,
                push_z_max=cfg.push_z_max,
            )
        )
        self._window: Any = None  # 由 start_collection 创建（延迟 import pygame）

    # ── 采集会话钩子（由采集脚本统一调用） ──────────────

    def start_collection(self) -> None:
        """启动 pygame 2D 俯视遥操作窗口并打印按键说明。"""
        if not os.environ.get("DISPLAY"):
            raise RuntimeError(
                "PushT 鼠标遥操作需要图形界面：未检测到 DISPLAY。"
                "请在有图形界面的环境运行。"
            )
        from .push_t_teleop import TeleopWindow

        self._window = TeleopWindow(self._state)
        self._window.start()
        print("\n" + "=" * 60)
        print("PushT 鼠标遥操作采集")
        print("  左键 = 切换鼠标控制   右键 = 开始/结束并保存录制")
        print("  中键 = 丢弃当前集     Esc = 退出")
        print("  滚轮 = TCP 升降   [ ] = 鼠标灵敏度   - = = 滚轮灵敏度")
        print("  （期望 TCP 为黄色十字；TCP z 落在可推动带内时状态栏显示 PUSH OK）")
        print("=" * 60)

    def end_collection(self) -> None:
        """关闭遥操作窗口（幂等）。"""
        if self._window is not None:
            self._window.close()
            self._window = None

    # ── 名称 ──────────────────────────────────────────

    def prefixed_ee_site(self) -> str:
        return f"{self.prefix}{self._cfg.ee_site}"

    def prefixed_object(self, name: str) -> str:
        return f"{self.prefix}{name}"

    # ── 生命周期 ───────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        # 新 episode 开始：期望 TCP 锚定到当前 EE（窗口初始十字 = 真实 TCP）
        ee = self.get_ee_pose(self.prefixed_ee_site())
        self._state.anchor_desired(ee[:3])

    # ── 遥操作控制 ────────────────────────────────────

    def publish_teleop_state(self, recording: bool) -> None:
        """把最新物理状态回写 TeleopState（供窗口渲染 / 状态栏）。"""
        tcp = self.get_ee_pose(self.prefixed_ee_site())[:3]
        tobj = self.get_object_pose(self.prefixed_object(self._cfg.t_obj))
        tgt = self.get_object_pose(self.prefixed_object(self._cfg.t_target))
        pushable = self._cfg.push_z_min <= tcp[2] <= self._cfg.push_z_max
        self._state.publish(
            tcp=tcp,
            t_obj_pos=tobj[:3],
            t_obj_yaw=_quat_to_yaw(tobj[3:7]),
            target_pos=tgt[:3],
            sim_time=float(self.data.time),
            recording=bool(recording),
            pushable=bool(pushable),
        )

    # ── 录制控制（由 run_episode 每策略步调用） ─────────

    def recording_decision(self, recording: bool) -> RecordingDecision | None:
        """鼠标遥操作录制控制：先回写物理状态供窗口渲染，再消费鼠标事件。

        - 未录制（等待就位）：左键右键（consume_start）→ START 开始录制；
          Esc → QUIT 退出
        - 录制中：右键（consume_stop）→ SAVED 保存；中键（consume_discard）
          → DISCARDED 丢弃；Esc → QUIT 退出
        """
        self.publish_teleop_state(recording)
        if not recording:
            if self._state.consume_start():
                return RecordingDecision.START
            if self._state.consume_quit():
                return RecordingDecision.QUIT
        else:
            if self._state.consume_stop():
                return RecordingDecision.SAVED
            if self._state.consume_discard():
                return RecordingDecision.DISCARDED
            if self._state.consume_quit():
                return RecordingDecision.QUIT
        return None

    # ── Controller 接口 ───────────────────────────────

    def step(self) -> dict[str, np.ndarray]:
        self.current_step += 1
        cfg = self._cfg
        desired = self._state.desired()
        target_pos = np.array([
            float(np.clip(desired[0], cfg.workspace_x[0], cfg.workspace_x[1])),
            float(np.clip(desired[1], cfg.workspace_y[0], cfg.workspace_y[1])),
            float(np.clip(desired[2], cfg.tcp_z_min, cfg.tcp_z_max)),
        ])
        return {
            self.prefix: self.make_action(
                target_pos, self._grasp_quat, gripper_cmd=cfg.gripper_cmd,
            )
        }

    # ── 成功判定（评估环境使用） ──────────────────────

    def check_success(self) -> bool:
        tobj = self.get_object_pose(self.prefixed_object(self._cfg.t_obj))
        tgt = self.get_object_pose(self.prefixed_object(self._cfg.t_target))
        dist = float(np.linalg.norm(tobj[:2] - tgt[:2]))
        d_yaw = abs((_quat_to_yaw(tobj[3:7]) - _quat_to_yaw(tgt[3:7]) + np.pi)
                    % (2 * np.pi) - np.pi)
        return dist < self._cfg.success_dist and d_yaw < self._cfg.success_yaw
