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
from mujoco_lerobot.configs.config_loader import RobotConfig
from mujoco_lerobot.configs.dataset_config import DatasetConfig


TASK_CONFIG = "configs/tasks/tasks.yaml"


def test_load_tasks():
    tasks = load_tasks(TASK_CONFIG)
    assert "pick_place" in tasks
    assert "dual_pick_place" in tasks
    assert "push_t" in tasks
    assert tasks["pick_place"].scene_config_file  # 场景配置连接
    assert tasks["pick_place"].teacher_config_file  # teacher 配置连接


def test_get_task_list():
    tl = get_task_list(TASK_CONFIG)
    assert "pick_place" in tl
    assert "dual_pick_place" in tl
    assert "push_t" in tl


def test_load_scene_config_pick_place():
    sc = load_scene_config("pick_place", TASK_CONFIG)
    assert len(sc.robots) == 1
    assert sc.robots[0].prefix == ""
    assert sc.action_dim == 7
    # 当前场景：全局相机 + 手眼相机（2 个）
    assert len(sc.cameras) == 2
    assert {c.name for c in sc.cameras} == {
        "global_realsense_link_CAMERA",
        "realsense_link_CAMERA",
    }
    for c in sc.cameras:
        assert c.width == 640
        assert c.height == 480


def test_load_scene_config_dual():
    sc = load_scene_config("dual_pick_place", TASK_CONFIG)
    assert len(sc.robots) == 2
    assert sc.robots[0].prefix == "A_"
    assert sc.action_dim == 14
    assert len(sc.cameras) == 3
    # 双臂 ik_solver 配置（vel_limit 显式写）
    for r in sc.robots:
        assert r.ik_solver.vel_limit == [3.1416] * 6
        assert r.ik_solver.collision_avoidance.enabled is False


def test_robot_ik_solver_defaults():
    """未配置 ik_solver 时使用与旧硬编码一致的默认值。"""
    r = RobotConfig.from_dict({"name": "ur5e"})
    ik = r.ik_solver
    assert ik.vel_limit == [3.1416] * 6
    assert ik.pos_cost == 1.0
    assert ik.ori_cost == 1.0
    assert ik.gain == 1.0
    assert ik.lm_damping == 1e-6
    assert ik.posture_cost == 1e-3
    assert ik.solver == "daqp"
    assert ik.damping == 1e-12
    assert ik.safety_break is False
    assert ik.max_iters == 1
    assert ik.pos_threshold == 0.01
    assert ik.ori_threshold == 0.1
    ca = ik.collision_avoidance
    assert ca.enabled is False
    assert ca.gain == 0.85
    assert ca.minimum_distance == 0.005
    assert ca.detection_distance == 0.01
    assert ca.bound_relaxation == 0.0
    assert ca.broadphase is True
    assert ca.pairs == []


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


def test_scene_config_push_t():
    sc = load_scene_config("push_t", TASK_CONFIG)
    assert len(sc.robots) == 1
    assert sc.robots[0].prefix == ""
    assert sc.action_dim == 7
    assert len(sc.cameras) == 2
    assert {c.name for c in sc.cameras} == {
        "global_realsense_link_CAMERA",
        "realsense_link_CAMERA",
    }
    # default_qpos 应使 TCP 初始落在可推动高度带内（见场景注释）
    assert sc.robots[0].default_qpos[0] == 3.14159


def test_teacher_config_push_t():
    tc = load_teacher_config("configs/teachers/push_t.yaml")
    assert tc.type == "PushTTeacher"
    assert tc.t_obj == "t_obj"
    assert tc.t_target == "t_target"
    assert tc.workspace_x == (-0.6, 0.6)
    assert tc.workspace_y == (-0.45, 0.45)
    assert tc.tcp_z_min == 0.62 and tc.tcp_z_max == 0.85
    assert tc.push_z_min == 0.66 and tc.push_z_max == 0.72
    assert tc.sens_mouse == 1.5 and tc.sens_wheel == 0.008
    assert tc.success_dist == 0.05 and tc.success_yaw == 0.35


def test_dataset_config_push_t():
    dc = DatasetConfig.from_yaml("configs/dataset/dataset_push_t.yaml")
    assert dc.use_recode_scale is False
    assert dc.recode_hz == 100.0
    assert dc.max_scale == 1
    assert dc.state_dim == 7
    assert dc.action_dim == 7
    assert dc.flat_mode is True
    assert dc.depth_range == (0.1, 2.0)
    assert dc.camera_config_file.endswith("scene_push_t.yaml")


def test_dataset_camera_crf_defaults(tmp_path):
    """未指定 depth_crf/rgb_crf 时：视频编码走默认（深度无损、rgb CRF=30）。"""
    import yaml
    from pathlib import Path

    raw = yaml.safe_load(Path("configs/dataset/dataset_pick_place.yaml").read_text())
    cam = raw["state"]["camera"]
    cam.pop("depth_crf", None)
    cam.pop("rgb_crf", None)
    p = tmp_path / "ds_default.yaml"
    p.write_text(yaml.safe_dump(raw))
    dc = DatasetConfig.from_yaml(str(p))
    assert dc.depth_crf is None
    assert dc.rgb_crf is None

    from mujoco_lerobot.data.dataset_writer import LeRobotDatasetConfig, build_video_encoders

    wcfg = LeRobotDatasetConfig(repo_id="x", root="/tmp/x", fps=100, vcodec="hevc", preset=None, g=2)
    depth_enc, rgb_enc = build_video_encoders(wcfg, dc)
    # 深度默认无损（lossless=1）
    assert "lossless=1" in depth_enc.extra_options.get("x265-params", "")
    # rgb 默认 CRF=30
    assert rgb_enc.crf == 30


def test_dataset_camera_crf_pick_place_config():
    """pick_place 数据集配置当前显式指定了画质 CRF。"""
    dc = DatasetConfig.from_yaml("configs/dataset/dataset_pick_place.yaml")
    assert dc.depth_crf == 15
    assert dc.rgb_crf == 20


def test_dataset_camera_crf_override(tmp_path):
    """设置 depth_crf/rgb_crf 后：深度去无损改有损 HEVC（按 CRF），rgb 用指定 CRF。"""
    import yaml
    from pathlib import Path

    raw = yaml.safe_load(Path("configs/dataset/dataset_pick_place.yaml").read_text())
    cam = raw["state"]["camera"]
    cam["depth_crf"] = 30
    cam["rgb_crf"] = 18
    p = tmp_path / "ds_crf.yaml"
    p.write_text(yaml.safe_dump(raw))

    dc = DatasetConfig.from_yaml(str(p))
    assert dc.depth_crf == 30
    assert dc.rgb_crf == 18

    from mujoco_lerobot.data.dataset_writer import LeRobotDatasetConfig, build_video_encoders

    wcfg = LeRobotDatasetConfig(repo_id="x", root="/tmp/x", fps=100, vcodec="hevc", preset=None, g=2)
    depth_enc, rgb_enc = build_video_encoders(wcfg, dc)
    # 深度：去掉 lossless、改用有损 HEVC（crf=30 生效）
    assert "lossless" not in depth_enc.extra_options.get("x265-params", "")
    assert depth_enc.crf == 30
    # rgb：使用指定 CRF
    assert rgb_enc.crf == 18
