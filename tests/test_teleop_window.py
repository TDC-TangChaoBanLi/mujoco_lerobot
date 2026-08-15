"""PushT 遥操作（TeleopState / PushTTeacher）无头测试。

TeleopState 不依赖 pygame，可完全无头验证：
锚定 / 鼠标增量 / 滚轮 / 灵敏度 / 事件 consume / 屏幕变换；
PushTTeacher 在真实模型上验证 step 输出、publish 回写与 check_success；
ScriptedTeacherController + teacher_kwargs 透传冒烟。
"""

from __future__ import annotations

import threading
import time

import mujoco
import numpy as np
import pytest

from mujoco_lerobot.configs import load_scene_config, load_teacher_config
from mujoco_lerobot.configs.dataset_config import DatasetConfig
from mujoco_lerobot.data.teachers.push_t_teleop import TeleopState, TeleopParams


# ═══════════════════════════════════════════════════════
# TeleopState 纯逻辑
# ═══════════════════════════════════════════════════════

def make_state(**kw) -> TeleopState:
    params = TeleopParams(**kw) if kw else TeleopParams()
    return TeleopState(params)


def test_state_initial_desired():
    s = make_state()
    d = s.desired()
    assert d[2] == pytest.approx(s.params.tcp_z_min + 0.05)


def test_anchor_desired():
    s = make_state()
    s.anchor_desired(np.array([0.3, -0.2, 9.0]))  # z 超上限 → clip
    d = s.desired()
    assert d[0] == pytest.approx(0.3)
    assert d[1] == pytest.approx(-0.2)
    assert d[2] == pytest.approx(s.params.tcp_z_max)


def test_toggle_control_anchors_and_message():
    s = make_state()
    s.publish(
        tcp=np.array([0.1, 0.2, 0.7]),
        t_obj_pos=np.array([0.3, 0.0, 0.63]),
        t_obj_yaw=0.0,
        target_pos=np.array([0.4, 0.15, 0.62]),
        sim_time=0.0,
        recording=False,
        pushable=True,
    )  # 先 publish（仿真线程已回写物理状态）
    s.toggle_control(np.array([0.1, 0.2, 0.7]))
    assert s.snapshot().control is True
    d = s.desired()
    assert d[:2] == pytest.approx([0.1, 0.2])
    assert "ON" in s.snapshot().message
    s.toggle_control()
    assert s.snapshot().control is False
    assert "OFF" in s.snapshot().message


def test_mouse_delta_moves_desired_and_clips():
    s = make_state()
    s.publish(
        tcp=np.array([0.0, 0.0, 0.68]),
        t_obj_pos=np.array([0.3, 0.0, 0.63]),
        t_obj_yaw=0.0,
        target_pos=np.array([0.4, 0.15, 0.62]),
        sim_time=0.0,
        recording=False,
        pushable=True,
    )
    s.toggle_control(np.array([0.0, 0.0, 0.68]))
    s.apply_mouse_delta(100, 0)   # 右移 → x 增
    d1 = s.desired()
    assert d1[0] > 0
    s.apply_mouse_delta(0, 100)   # 下移 → 世界 y 减小（屏幕 v 翻转）
    d2 = s.desired()
    assert d2[1] < 0
    # 大幅移动 → 裁剪到工作区
    s.apply_mouse_delta(1e6, 0)
    assert s.desired()[0] == pytest.approx(s.params.workspace_x[1])
    s.apply_mouse_delta(0, 1e6)
    assert s.desired()[1] == pytest.approx(s.params.workspace_y[0])


def test_mouse_delta_ignored_when_not_controlling():
    s = make_state()
    s.apply_mouse_delta(100, 0)
    assert s.desired()[0] == pytest.approx(0.0)


def test_mouse_delta_invert_flips_signs():
    s = make_state(invert_mouse_x=True, invert_mouse_y=True)
    s.publish(
        tcp=np.array([0.0, 0.0, 0.68]),
        t_obj_pos=np.array([0.3, 0.0, 0.63]),
        t_obj_yaw=0.0,
        target_pos=np.array([0.4, 0.15, 0.62]),
        sim_time=0.0,
        recording=False,
        pushable=True,
    )
    s.toggle_control(np.array([0.0, 0.0, 0.68]))
    s.apply_mouse_delta(100, 0)   # 右移（物理）→ 反转后 x 减
    assert s.desired()[0] < 0
    s.apply_mouse_delta(0, 100)   # 下移（物理）→ 反转后 y 增
    assert s.desired()[1] > 0
    # 默认（标准）状态方向不受影响：右移 → x 增
    s2 = make_state()
    s2.publish(
        tcp=np.array([0.0, 0.0, 0.68]),
        t_obj_pos=np.array([0.3, 0.0, 0.63]),
        t_obj_yaw=0.0,
        target_pos=np.array([0.4, 0.15, 0.62]),
        sim_time=0.0,
        recording=False,
        pushable=True,
    )
    s2.toggle_control(np.array([0.0, 0.0, 0.68]))
    s2.apply_mouse_delta(100, 0)
    assert s2.desired()[0] > 0


def test_wheel_scroll_changes_z_and_clips():
    s = make_state()
    s.wheel_scroll(10)
    assert s.desired()[2] > s.params.tcp_z_min + 0.05
    s.wheel_scroll(-100)
    assert s.desired()[2] == pytest.approx(s.params.tcp_z_min)
    s.wheel_scroll(100)
    assert s.desired()[2] == pytest.approx(s.params.tcp_z_max)


def test_sens_adjust():
    s = make_state()
    s.adjust_mouse_sens(0.2)
    assert s.snapshot().mouse_sens == pytest.approx(s.params.mouse_sens + 0.2)
    # 滚轮灵敏度为乘法调节（×(1+delta)），限制在 [0.001, 0.1]
    s.adjust_wheel_sens(0.1)
    assert s.snapshot().wheel_sens == pytest.approx(s.params.wheel_sens * 1.1)
    s.adjust_wheel_sens(-0.1)
    assert s.snapshot().wheel_sens == pytest.approx(s.params.wheel_sens * 1.1 * 0.9)
    s.adjust_wheel_sens(-10)   # 下限
    assert s.snapshot().wheel_sens == pytest.approx(0.001)
    s.adjust_wheel_sens(100)   # 上限
    assert s.snapshot().wheel_sens == pytest.approx(0.1)


def test_toggle_control_keeps_z_when_anchor_z_invalid():
    """未 publish（tcp 全零）时 toggle_control 不应把黄十字锚定到固定点。"""
    s = make_state()
    d_before = s.desired().copy()
    s.toggle_control(np.zeros(3))  # 模拟 tcp 未 publish（全零）
    assert s.snapshot().control is True
    d = s.desired()
    # 完全不锚定：x/y/z 全部保持（黄十字不跳变）
    assert d == pytest.approx(d_before)
    # 已 publish 后：xy 锚定；z 仍只接受合理高度
    s.toggle_control()  # OFF
    s.publish(
        tcp=np.array([0.1, 0.2, 0.7]),
        t_obj_pos=np.array([0.3, 0.0, 0.63]),
        t_obj_yaw=0.0,
        target_pos=np.array([0.4, 0.15, 0.62]),
        sim_time=0.0,
        recording=False,
        pushable=True,
    )
    s.toggle_control(np.array([0.1, 0.2, 0.7]))
    assert s.desired()[:2] == pytest.approx([0.1, 0.2])
    assert s.desired()[2] == pytest.approx(0.7)
    s.toggle_control()  # OFF
    s.toggle_control(np.array([0.3, 0.4, 0.0]))  # z=0 不在合理范围 → 不覆盖 z
    assert s.desired()[:2] == pytest.approx([0.3, 0.4])
    assert s.desired()[2] == pytest.approx(0.7)


def test_t_geometry_matches_xml_and_rotates_around_origin():
    """T 两段矩形几何与 t_obj.xml 一致，且整体绕 T 原点旋转（修复 cross 绕自身旋转）。"""
    from mujoco_lerobot.data.teachers.push_t_teleop import _t_local_points

    # yaw=0：stem 中心 (0,0)，cross 中心 (0, 0.045)
    stem = np.mean(_t_local_points(0.0, 0), axis=0)
    cross = np.mean(_t_local_points(0.0, 1), axis=0)
    assert stem == pytest.approx([0.0, 0.0])
    assert cross == pytest.approx([0.0, 0.045])
    # 半尺寸（与 t_obj.xml：stem size 0.01×0.035，cross size 0.04×0.01 中心偏移 0.045）
    stem_pts = np.asarray(_t_local_points(0.0, 0))
    assert stem_pts[:, 0].max() == pytest.approx(0.01)
    assert stem_pts[:, 1].max() == pytest.approx(0.035)
    cross_pts = np.asarray(_t_local_points(0.0, 1))
    assert cross_pts[:, 0].max() == pytest.approx(0.04)
    assert cross_pts[:, 1].max() == pytest.approx(0.055)
    # yaw=π/2：整体绕原点旋转 → cross 中心转到 (-0.045, 0)（逆时针为正）
    cross90 = np.mean(_t_local_points(np.pi / 2, 1), axis=0)
    assert cross90 == pytest.approx([-0.045, 0.0], abs=1e-9)


def test_events_consume_once():
    s = make_state()
    s.request_start()
    s.request_stop()
    s.request_discard()
    s.request_quit()
    assert s.consume_start() is True
    assert s.consume_start() is False  # 一次性
    assert s.consume_stop() is True
    assert s.consume_stop() is False
    assert s.consume_discard() is True
    assert s.consume_quit() is True


def test_publish_and_snapshot():
    s = make_state()
    s.publish(
        tcp=np.array([0.3, 0.1, 0.68]),
        t_obj_pos=np.array([0.4, 0.0, 0.63]),
        t_obj_yaw=1.2,
        target_pos=np.array([0.5, 0.15, 0.62]),
        sim_time=3.5,
        recording=True,
        pushable=True,
    )
    snap = s.snapshot()
    assert snap.tcp[0] == pytest.approx(0.3)
    assert snap.t_obj_yaw == pytest.approx(1.2)
    assert snap.recording is True and snap.pushable is True
    assert snap.sim_time == pytest.approx(3.5)


def test_screen_transform():
    s = make_state()
    scale, u0, v0 = s.screen_transform()
    # 工作区左下角 → 屏幕左下角（留边内）
    u, v = s.world_to_screen(s.params.workspace_x[0], s.params.workspace_y[0])
    assert u == pytest.approx(u0)
    assert v == pytest.approx(v0 + (s.params.workspace_y[1] - s.params.workspace_y[0]) * scale)
    # 屏幕顶部 y 更大（v 翻转）：世界 y 增大 → v 减小
    u1, v1 = s.world_to_screen(0.0, s.params.workspace_y[1])
    u2, v2 = s.world_to_screen(0.0, s.params.workspace_y[0])
    assert v1 < v2
    # 全部落在窗口内
    assert u0 >= 0 and v0 >= 0
    assert u <= s.params.window_width and v <= s.params.window_height


# ═══════════════════════════════════════════════════════
# PushTTeacher（真实模型）
# ═══════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def push_env():
    sc = load_scene_config("push_t", "configs/tasks/tasks.yaml")
    dc = DatasetConfig.from_yaml("configs/dataset/dataset_push_t.yaml")
    from mujoco_lerobot.data import SimulationManager

    mgr = SimulationManager(sc, dc, render=False)
    # 应用 default_qpos（scene yaml）并初始化 xpos/xmat
    mgr._reset_mgr.reset(randomize_objects=False)
    mujoco.mj_forward(mgr.model, mgr.data)
    yield sc, dc, mgr
    mgr.close()


def test_push_teacher_step_shape(push_env):
    sc, dc, mgr = push_env
    tc = load_teacher_config(sc.task.teacher_config_file)
    from mujoco_lerobot.data.teachers import create_teacher

    t = create_teacher(tc.type, mgr.model, mgr.data, config=tc)
    t.reset()
    out = t.step()
    assert out[""].shape == (8,)
    assert np.isfinite(out[""]).all()
    # 姿态固定夹爪朝下（grasp_quat），动作含 gripper
    assert out[""][3:7] == pytest.approx(np.asarray(tc.grasp_quat))
    assert out[""][7] == pytest.approx(tc.gripper_cmd)


def test_push_teacher_desired_clip(push_env):
    sc, dc, mgr = push_env
    tc = load_teacher_config(sc.task.teacher_config_file)
    from mujoco_lerobot.data.teachers import create_teacher

    t = create_teacher(tc.type, mgr.model, mgr.data, config=tc)
    t.reset()
    # 期望 TCP 推到工作区外 → step 应裁剪
    t._state.anchor_desired(np.array([9.0, 9.0, 0.68]))
    out = t.step()[""]
    assert out[0] <= tc.workspace_x[1]
    assert out[1] <= tc.workspace_y[1]


def test_push_teacher_publish_and_check_success(push_env):
    sc, dc, mgr = push_env
    tc = load_teacher_config(sc.task.teacher_config_file)
    from mujoco_lerobot.data.teachers import create_teacher

    t = create_teacher(tc.type, mgr.model, mgr.data, config=tc)
    t.reset()
    t.publish_teleop_state(recording=True)
    snap = t._state.snapshot()
    # 初始 T 静置桌面，未推动 → 不成功；但初始位姿应在可推动带内
    assert t.check_success() is False
    assert snap.t_obj_pos[2] == pytest.approx(0.63, abs=0.05)
    assert snap.pushable is True  # default_qpos 使 TCP 初始 z 落在可推动带

    # 人为把 T 移到目标位 → 判成功（freejoint 在 mount body 上）
    mj_t_tgt = t.prefixed_object(tc.t_target)
    tgt = t.get_object_pose(mj_t_tgt)
    mount = t.prefixed_object(tc.t_obj) + "_mount"
    jid = mgr.model.body(mount).jntadr[0]
    adr = mgr.model.jnt_qposadr[jid]
    q = mgr.data.qpos[adr : adr + 7].copy()
    q[0:3] = tgt[0:3]
    q[3:7] = tgt[3:7]
    mgr.data.qpos[adr : adr + 7] = q
    mujoco.mj_forward(mgr.model, mgr.data)
    assert t.check_success() is True


def test_controller_teacher_kwargs_passthrough(push_env):
    """ScriptedTeacherController 通过 teacher_kwargs 注入共享 TeleopState。"""
    sc, dc, mgr = push_env
    tc = load_teacher_config(sc.task.teacher_config_file)
    from mujoco_lerobot.data import ScriptedTeacherController

    shared = TeleopState(TeleopParams(
        window_width=tc.window_width,
        window_height=tc.window_height,
        workspace_x=tc.workspace_x,
        workspace_y=tc.workspace_y,
        mouse_sens=tc.sens_mouse,
        wheel_sens=tc.sens_wheel,
        sens_step=tc.sens_step,
        tcp_z_min=tc.tcp_z_min,
        tcp_z_max=tc.tcp_z_max,
        push_z_min=tc.push_z_min,
        push_z_max=tc.push_z_max,
    ))
    ctrl = ScriptedTeacherController(
        sc, tc, mgr.model, mgr.data, teacher_kwargs={"state": shared}
    )
    teacher = next(iter(ctrl._teachers.values()))
    assert teacher._state is shared  # 注入生效，而非自建


def test_run_episode_teleop_save_and_discard(push_env):
    """run_episode + PushTTeacher.recording_decision：saved / discarded 两路径。

    录制生命周期完全由 teacher 的 recording_decision 控制（无头、无窗口）：
    - 等待阶段（未录制）：teacher 每策略步 publish 物理状态并消费鼠标事件
    - 用户右键（request_start）→ START 开始录制；右键（request_stop）→ SAVED
    - 中键（request_discard）→ DISCARDED
    """
    sc, dc, mgr = push_env
    tc = load_teacher_config(sc.task.teacher_config_file)
    from mujoco_lerobot.data import ScriptedTeacherController

    ctrl = ScriptedTeacherController(sc, tc, mgr.model, mgr.data)
    teacher = next(iter(ctrl._teachers.values()))
    state = teacher._state

    # 保存：0.15s 后开始，0.8s 后结束
    def user():
        time.sleep(0.15)
        state.request_start()
        time.sleep(0.8)
        state.request_stop()

    threading.Thread(target=user, daemon=True).start()
    result, n = mgr.run_episode(ctrl)
    assert result == "saved"
    assert n > 0

    # 丢弃：开始后中键丢弃
    def user2():
        time.sleep(0.15)
        state.request_start()
        time.sleep(0.5)
        state.request_discard()

    threading.Thread(target=user2, daemon=True).start()
    result2, n2 = mgr.run_episode(ctrl)
    assert result2 == "discarded"
