"""配置加载测试 — 任务/场景/数据集/teacher 配置。"""

from __future__ import annotations

import numpy as np
import pytest

from mujoco_lerobot.configs import (
    load_scene_config,
    load_tasks,
    get_task_list,
    load_teacher_config,
)
from mujoco_lerobot.configs.dataset_config import DatasetConfig


TASK_CONFIG = "configs/tasks/tasks.yaml"


def test_load_tasks():
    tasks = load_tasks(TASK_CONFIG)
    assert "pick_place" in tasks
    assert "dual_pick_place" in tasks
    assert tasks["pick_place"].scene_config_file  # 场景配置连接
    assert tasks["pick_place"].teacher_config_file  # teacher 配置连接


def test_get_task_list():
    tl = get_task_list(TASK_CONFIG)
    assert "pick_place" in tl
    assert "dual_pick_place" in tl


def test_load_scene_config_pick_place():
    sc = load_scene_config("pick_place", TASK_CONFIG)
    assert len(sc.robots) == 1
    assert sc.robots[0].prefix == ""
    assert sc.action_dim == 7
    assert len(sc.cameras) == 1
    assert sc.cameras[0].width == 640
    assert sc.cameras[0].height == 480


def test_load_scene_config_dual():
    sc = load_scene_config("dual_pick_place", TASK_CONFIG)
    assert len(sc.robots) == 2
    assert sc.robots[0].prefix == "A_"
    assert sc.action_dim == 14
    assert len(sc.cameras) == 3


def test_dataset_config_pick_place():
    dc = DatasetConfig.from_yaml("configs/dataset/dataset_pick_place.yaml")
    # 当前单臂配置为 use_recode_scale=false（扁平 / ACT 兼容）
    assert dc.use_recode_scale is False
    assert dc.recode_hz == 100.0
    assert dc.max_scale == 1
    # state: joint.position (1,7)
    state = dc.state_sources()
    assert len(state) == 1
    assert state[0].num_subs == 1
    assert state[0].dim_per_sub == 7
    # action: (1,7)
    assert dc.action_scale == 1
    assert dc.action_total_dim_per_sub() == 7
    assert dc.state_dim == 7
    assert dc.action_dim == 7
    # 扁平模式产出 LeRobot 标准 features（ACT 兼容）
    assert dc.flat_mode is True
    feats = dc.build_features([])
    assert feats["observation.state"]["shape"] == (7,)
    assert feats["action"]["shape"] == (7,)


def test_dataset_config_pick_place_multirate():
    """多速率模式（use_recode_scale=true）的解析。"""
    dc = DatasetConfig.from_yaml("configs/dataset/dataset_pick_place.yaml")
    assert dc.flat_mode is True
    # 切换为多速率后，state 源应展开为 (R, D)
    dc.use_recode_scale = True
    assert dc.flat_mode is False


def test_dataset_config_dual():
    dc = DatasetConfig.from_yaml("configs/dataset/dataset_dual_pick_place.yaml")
    assert dc.use_recode_scale is True
    # state: joint(3,14) + force(3,6) + torque(3,6) = 42+18+18=78
    assert dc.state_dim == 78
    assert dc.action_dim == 42
    # 深度范围在 dataset 配置的 state.camera 下定义
    assert dc.depth_range == (0.1, 2.0)
    assert dc.camera_config_file.endswith("scene_dual_pick_place.yaml")


def test_dataset_config_use_recode_scale_false(tmp_path):
    import yaml
    from pathlib import Path
    raw = yaml.safe_load(Path("configs/dataset/dataset_pick_place.yaml").read_text())
    raw["use_recode_scale"] = False
    p = tmp_path / "ds.yaml"
    p.write_text(yaml.safe_dump(raw))
    dc = DatasetConfig.from_yaml(str(p))
    assert dc.use_recode_scale is False
    assert dc.max_scale == 1
    for s in dc.sources:
        assert s.num_subs == 1
    assert dc.state_dim == 7
    assert dc.action_dim == 7
    assert dc.flat_mode is True


def test_dataset_config_new_source_types(tmp_path):
    """新增源类型（joint velocity/effort、sensor gyro/accel、frame、action tcp）解析。"""
    import yaml
    from pathlib import Path
    raw = yaml.safe_load(Path("configs/dataset/dataset_pick_place.yaml").read_text())
    raw["use_recode_scale"] = True
    # state 增加 velocity/effort 与 frame 位姿
    raw["state"]["joint"]["velocity"] = {"joint_names": ["ur_shoulder_pan_joint"]}
    raw["state"]["joint"]["effort"] = {"joint_names": ["ur_shoulder_pan_joint"]}
    raw["state"]["frame"] = {
        "position": {"frame_site": "_tcp", "site_names": ["_tcp"]},
        "quat": {"frame_site": "world", "site_names": ["_tcp"]},
    }
    # action 增加 tcp 位姿（relative / absolute）
    raw["action"]["tcp"] = {
        "position": {"type": "relative", "frame_site": "world",
                      "site_names": ["_tcp"]},
        "euler": {"type": "absolute", "site_names": ["_tcp"]},
    }
    p = tmp_path / "ds2.yaml"
    p.write_text(yaml.safe_dump(raw))
    dc = DatasetConfig.from_yaml(str(p))
    types = {s.name: s.source_type for s in dc.sources}
    assert types["state.joint.velocity"] == "joint_vel"
    assert types["state.joint.effort"] == "joint_effort"
    assert types["state.frame.position"] == "site_position"
    assert types["state.frame.quat"] == "site_quat"
    assert types["action.tcp.position"] == "site_position"
    assert types["action.tcp.euler"] == "site_euler"
    by_name = {s.name: s for s in dc.sources}
    assert by_name["state.frame.position"].frame_site == "_tcp"
    assert by_name["state.frame.quat"].frame_site == "world"
    assert by_name["action.tcp.position"].pose_type == "relative"
    assert by_name["action.tcp.euler"].pose_type == "absolute"


def test_teacher_configs():
    sc = load_scene_config("pick_place", TASK_CONFIG)
    tc = load_teacher_config(sc.task.teacher_config_file)
    assert tc.type == "PickPlaceTeacher"
    assert tc.thresh.max_retries == 3

    sc2 = load_scene_config("dual_pick_place", TASK_CONFIG)
    tc2 = load_teacher_config(sc2.task.teacher_config_file)
    assert tc2.type == "DualPickPlaceTeacher"
    assert tc2.grasp_euler_a is not None
