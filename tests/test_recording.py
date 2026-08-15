"""录制决策（RecordingDecision）纯逻辑测试。

验证 run_episode 的录制生命周期契约：
- RecordingDecision 枚举值语义
- Teacher 基类默认 recording_decision（scripted 自动 teacher）：
  未录制自动 START；状态机跑完（is_done）时按 check_success 判定 SAVED/DISCARDED
"""

from __future__ import annotations

from mujoco_lerobot.data.recording import RecordingDecision
from mujoco_lerobot.data.teachers.base import Teacher, TeacherState


class _DummyTeacher(Teacher):
    """最小 Teacher 子类：跳过模型依赖，仅测录制决策逻辑。"""

    def __init__(self, success: bool = True) -> None:
        self.state = TeacherState.RUNNING
        self.current_step = 0
        self._success = success

    def step(self) -> dict:
        return {}

    def check_success(self) -> bool:
        return self._success


def test_enum_values():
    assert RecordingDecision.START.value == "start"
    assert RecordingDecision.SAVED.value == "saved"
    assert RecordingDecision.DISCARDED.value == "discarded"
    assert RecordingDecision.QUIT.value == "quit"


def test_teacher_default_auto_start():
    """未录制时默认返回 START（scripted 自动 teacher 立即开始录制）。"""
    t = _DummyTeacher()
    assert t.recording_decision(False) is RecordingDecision.START


def test_teacher_default_success_saved():
    """状态机跑完且物理成功 → SAVED。"""
    t = _DummyTeacher(success=True)
    t.state = TeacherState.SUCCESS
    assert t.recording_decision(True) is RecordingDecision.SAVED


def test_teacher_default_failure_discarded():
    """状态机跑完但物理未成功 → DISCARDED（丢弃重录）。"""
    t = _DummyTeacher(success=False)
    t.state = TeacherState.FAILURE
    assert t.recording_decision(True) is RecordingDecision.DISCARDED


def test_teacher_default_running_none():
    """录制中且状态机未跑完 → None（继续）。"""
    t = _DummyTeacher()
    assert t.recording_decision(True) is None


# ═══════════════════════════════════════════════════════
# 采集会话钩子（start_collection / end_collection / retry_limit）
# ═══════════════════════════════════════════════════════

def test_teacher_default_hooks_noop():
    """scripted teacher 默认钩子为 no-op，retry_limit 为 None。"""
    t = _DummyTeacher()
    t.start_collection()  # 不应抛异常
    t.end_collection()    # 不应抛异常（幂等）
    assert t.retry_limit is None


def test_teacher_hooks_called_once():
    """钩子调用顺序：start 在采集前、end 在采集后（模拟 collect_teacher 流程）。"""
    calls: list[str] = []

    class HookTeacher(_DummyTeacher):
        retry_limit = 1

        def start_collection(self) -> None:
            calls.append("start")

        def end_collection(self) -> None:
            calls.append("end")

    t = HookTeacher()
    t.start_collection()
    t.end_collection()
    assert calls == ["start", "end"]
    assert t.retry_limit == 1


def test_controller_hooks_aggregate_multi_arm():
    """ScriptedTeacherController 透传钩子并聚合 retry_limit。"""
    from mujoco_lerobot.data.controllers import ScriptedTeacherController

    class ArmA(_DummyTeacher):
        retry_limit = 2

    class ArmB(_DummyTeacher):
        retry_limit = 3

    ctrl = ScriptedTeacherController.__new__(ScriptedTeacherController)
    ctrl._multi_arm = False
    ctrl._teachers = {"a": ArmA(), "b": ArmB()}
    assert ctrl.retry_limit == 2  # 取最小非 None

    ctrl._teachers = {"a": ArmA(), "b": _DummyTeacher()}  # b 为 None
    assert ctrl.retry_limit == 2
