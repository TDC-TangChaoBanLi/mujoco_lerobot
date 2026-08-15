"""录制决策枚举 — 由 teacher 控制 episode 的录制生命周期。

``SimulationManager.run_episode`` 每策略步调用
``controller.recording_decision(recording)``，由 controller（teacher）决定
何时开始录制、何时结束、以及结束后 episode 的去留：

- ``START``     开始录制（从下一策略步起采样 / 渲染 / 记录帧）
- ``SAVED``     结束录制并保存本集（保留已写入的帧）
- ``DISCARDED`` 结束录制并丢弃本集（不保留）
- ``QUIT``      结束录制并退出整个采集流程

不同 teacher 的语义（示例）：
- scripted teacher（pick_place 等）：第一次策略步自动 START；状态机跑完时
  按 ``check_success()`` 返回 SAVED / DISCARDED。
- 遥操作 teacher（push_t 鼠标）：用户右键 START / 右键保存 / 中键丢弃 /
  Esc 退出。

独立模块，不依赖 controllers / teachers，避免双向导入。
"""

from __future__ import annotations

from enum import Enum


class RecordingDecision(str, Enum):
    """episode 录制决策（由 controller 每策略步给出）。"""

    START = "start"
    SAVED = "saved"
    DISCARDED = "discarded"
    QUIT = "quit"
