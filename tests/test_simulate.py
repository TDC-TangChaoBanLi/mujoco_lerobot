"""仿真底层测试 — MujocoWrapper / MinkIK / CameraRenderer / 采集器。"""

from __future__ import annotations

import numpy as np
import pytest

from mujoco_lerobot.configs import load_scene_config, load_teacher_config
from mujoco_lerobot.configs.dataset_config import DatasetConfig
from mujoco_lerobot.simulate import MujocoWrapper, MinkIK, CameraRenderer
from mujoco_lerobot.data.observation_collector import ObservationCollector


@pytest.fixture(scope="module")
def pick_place_env():
    sc = load_scene_config("pick_place", "configs/tasks/tasks.yaml")
    mj = MujocoWrapper(sc.task.scene_path)
    mj.open()
    yield sc, mj
    mj.close()


def test_mujoco_wrapper_joints(pick_place_env):
    sc, mj = pick_place_env
    r0 = sc.robots[0]
    q = mj.joint_qpos(r0.prefixed_arm_joints)
    assert q.shape == (6,)
    assert np.isfinite(q).all()


def test_mink_ik(pick_place_env):
    sc, mj = pick_place_env
    r0 = sc.robots[0]
    ik = MinkIK(
        mj.model, mj.data.qpos.copy(), dt=0.01,
        ee_site_name=r0.prefixed_ee_site,
        arm_joint_names=r0.prefixed_arm_joints,
    )
    cur = mj.joint_qpos(r0.prefixed_arm_joints)
    ee = mj.get_site_pose(r0.prefixed_ee_site)
    target = ee.copy()
    target[2] += 0.05
    jt = ik.solve(cur, target)
    assert jt.shape == (6,)
    assert np.isfinite(jt).all()


def test_camera_renderer(pick_place_env):
    sc, mj = pick_place_env
    mj.reset()
    for i, j in enumerate(sc.robots[0].prefixed_arm_joints):
        mj.set_joint_qpos(j, sc.robots[0].default_qpos[i])
    mj.forward()
    cr = CameraRenderer(sc.cameras, mj.model, mj.data)
    frames = cr.render_all(mj.data)
    assert len(frames) == 1
    f = frames[0]
    assert f.rgb.shape == (480, 640, 3)
    assert f.rgb.dtype == np.uint8
    assert f.depth.shape == (480, 640)
    assert f.depth.dtype == np.float32
    # 深度应包含近处物体（<2m 有占比），而非全远平面
    assert (f.depth < 2.0).mean() > 0.01
    cr.close()


def test_observation_collector_sampling(pick_place_env):
    sc, mj = pick_place_env
    mj.reset()
    dc = DatasetConfig.from_yaml("configs/dataset/dataset_pick_place.yaml")
    col = ObservationCollector(mj, dc)
    col.reset()
    for _ in range(dc.max_scale):
        col.sample()
    assert col.is_ready()
    frame = col.flush(task_id=0)
    assert "state.joint.position" in frame["state"]
    # 扁平模式：num_subs=1 → (1, 7)
    assert frame["state"]["state.joint.position"].shape == (1, 7)
    assert "action.joint.position" in frame["action"]


def test_observation_collector_multirate_sampling(pick_place_env, tmp_path):
    """多速率模式（use_recode_scale=true）的采样 shape 为 (R, D)。"""
    import yaml
    from pathlib import Path
    sc, mj = pick_place_env
    mj.reset()
    raw = yaml.safe_load(Path("configs/dataset/dataset_pick_place.yaml").read_text())
    raw["use_recode_scale"] = True
    raw["recode_hz"] = 30.0
    p = tmp_path / "ds_mr.yaml"
    p.write_text(yaml.safe_dump(raw))
    dc = DatasetConfig.from_yaml(str(p))
    assert dc.max_scale == 3
    col = ObservationCollector(mj, dc)
    col.reset()
    for _ in range(dc.max_scale):
        col.sample()
    assert col.is_ready()
    frame = col.flush(task_id=0)
    assert frame["state"]["state.joint.position"].shape == (3, 7)
    assert frame["action"]["action.joint.position"].shape == (3, 7)


def test_teacher_state_machine(pick_place_env):
    from mujoco_lerobot.data.controllers import ScriptedTeacherController

    sc, mj = pick_place_env
    mj.reset()
    tc = load_teacher_config(sc.task.teacher_config_file)
    ctrl = ScriptedTeacherController(sc, tc, mj.model, mj.data)
    ctrl.reset()
    obs = {"arm_joint_pos": mj.joint_qpos(sc.robots[0].prefixed_arm_joints)}
    action = ctrl.step(obs)
    assert action.shape == (7,)
    assert np.isfinite(action).all()


def test_teacher_registry_and_discovery():
    """teacher 注册表与自动发现：内置 teacher 均注册，配置可通过注册表加载。"""
    from mujoco_lerobot.data.teachers import (
        TEACHER_REGISTRY,
        TEACHER_CONFIG_REGISTRY,
        discover_teachers,
        create_teacher,
    )

    discover_teachers()
    assert "PickPlaceTeacher" in TEACHER_REGISTRY
    assert "DualPickPlaceTeacher" in TEACHER_REGISTRY
    assert TEACHER_CONFIG_REGISTRY["PickPlaceTeacher"].__name__ == "PickPlaceTeacherConfig"
    # 配置自动加载（通过注册表而非硬编码）
    tc = load_teacher_config("configs/teachers/pick_place.yaml")
    assert tc.type == "PickPlaceTeacher"


def test_teacher_check_success_pick_place(pick_place_env):
    """评估成功判定由 teacher 提供：pick_place 在 cube 放到 plate 上时判成功。"""
    from mujoco_lerobot.env.lerobot_env import MujocoLerobotEnv

    env = MujocoLerobotEnv(
        task_name="pick_place",
        dataset_config="configs/dataset/dataset_pick_place.yaml",
    )
    env.reset(seed=0)
    assert env._check_success() is False

    # 手动把 cube 放到 plate 上（freejoint qpos）
    mj = env._mj
    plate = mj.data.xpos[mj._body_ids["plate"]].copy()
    jid = mj.get_body_joint_id("cube_mount")
    adr = mj.model.jnt_qposadr[jid]
    mj.data.qpos[adr : adr + 3] = plate + np.array([0, 0, 0.03])
    mj.forward()
    assert env._check_success() is True
    env.close()


def test_env_render_fps_matches_recode_hz():
    """评估视频帧率 = recode_hz，使视频时长 = 仿真时长（无慢放）。"""
    from mujoco_lerobot.env.lerobot_env import MujocoLerobotEnv
    from mujoco_lerobot.configs.dataset_config import DatasetConfig

    env = MujocoLerobotEnv(
        task_name="pick_place",
        dataset_config="configs/dataset/dataset_pick_place.yaml",
    )
    dc = DatasetConfig.from_yaml("configs/dataset/dataset_pick_place.yaml")
    assert env.metadata["render_fps"] == int(round(dc.recode_hz))
    env.close()


def test_scene_view_config():
    """场景配置可指定 viewer 视角参数（评估/采集初始视角）。"""
    sc = load_scene_config("pick_place", "configs/tasks/tasks.yaml")
    assert sc.view.lookat == (0.45, 0.0, 0.65)
    assert sc.view.distance > 0
    assert sc.view.image_size == (640, 480)
    sc_dual = load_scene_config("dual_pick_place", "configs/tasks/tasks.yaml")
    assert sc_dual.view.distance > 0
    assert sc_dual.view.image_size == (640, 480)


def test_env_render_view():
    """eval 视频用 view 配置的自由视角渲染，且不改变观测相机视角。"""
    from mujoco_lerobot.env.lerobot_env import MujocoLerobotEnv

    env = MujocoLerobotEnv(
        task_name="pick_place",
        dataset_config="configs/dataset/dataset_pick_place.yaml",
    )
    env.reset(seed=0)
    vimg = env.render()
    assert vimg.shape == (480, 640, 3)
    assert vimg.dtype == np.uint8

    obs, _ = env.reset(seed=0)
    cam_rgb = obs["images.realsense_link_CAMERA.rgb"].transpose(1, 2, 0)
    # 视角不同 → 像素有差异；观测相机不受 render() 影响
    diff = np.abs(vimg.astype(int) - cam_rgb.astype(int)).mean()
    assert diff > 1.0
    env.close()
