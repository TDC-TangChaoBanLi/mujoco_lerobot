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


def test_mink_ik_ikconfig(pick_place_env):
    """传 ik_config（来自场景 yaml robot/ik_solver）构造并求解。"""
    sc, mj = pick_place_env
    r0 = sc.robots[0]
    ik = MinkIK(
        mj.model, mj.data.qpos.copy(), dt=0.01,
        ee_site_name=r0.prefixed_ee_site,
        arm_joint_names=r0.prefixed_arm_joints,
        ik_config=r0.ik_solver,
        prefix=r0.prefix,
    )
    cur = mj.joint_qpos(r0.prefixed_arm_joints)
    ee = mj.get_site_pose(r0.prefixed_ee_site)
    target = ee.copy()
    target[2] += 0.05
    jt = ik.solve(cur, target)
    assert jt.shape == (6,)
    assert np.isfinite(jt).all()
    # 场景配置的 vel_limit 应生效
    assert ik._vel_limit == [3.1416] * 6


def test_mink_ik_collision_avoidance(pick_place_env):
    """启用 CollisionAvoidanceLimit：构造成功 + 求解有限 + 内部多一个 limit。"""
    from mujoco_lerobot.configs.config_loader import (
        CollisionAvoidanceConfig,
        IKSolverConfig,
    )

    sc, mj = pick_place_env
    r0 = sc.robots[0]
    ik_cfg = IKSolverConfig(
        collision_avoidance=CollisionAvoidanceConfig(
            enabled=True,
            pairs=[(["ur_wrist_3_link"], ["table_surface"])],
        )
    )
    ik = MinkIK(
        mj.model, mj.data.qpos.copy(), dt=0.01,
        ee_site_name=r0.prefixed_ee_site,
        arm_joint_names=r0.prefixed_arm_joints,
        ik_config=ik_cfg,
    )
    # 简短名自动匹配 COLLISION_* geom + 精确名
    ids = MinkIK._resolve_geom_ids(mj.model, "", ["ur_wrist_3_link", "table_surface"])
    assert "COLLISION_ur_wrist_3_link_0" in {mj.model.geom(i).name for i in ids}
    assert "table_surface" in {mj.model.geom(i).name for i in ids}
    # ConfigurationLimit + VelocityLimit + CollisionAvoidanceLimit
    assert len(ik._limits) == 3

    cur = mj.joint_qpos(r0.prefixed_arm_joints)
    ee = mj.get_site_pose(r0.prefixed_ee_site)
    target = ee.copy()
    target[2] += 0.05
    jt = ik.solve(cur, target)
    assert jt.shape == (6,)
    assert np.isfinite(jt).all()


def test_mink_ik_collision_avoidance_bad_geom(pick_place_env):
    """配置了不存在的 geom 应抛出 ValueError 并提示。"""
    from mujoco_lerobot.configs.config_loader import (
        CollisionAvoidanceConfig,
        IKSolverConfig,
    )

    sc, mj = pick_place_env
    r0 = sc.robots[0]
    ik_cfg = IKSolverConfig(
        collision_avoidance=CollisionAvoidanceConfig(
            enabled=True,
            pairs=[(["nonexistent_geom"], ["table_surface"])],
        )
    )
    with pytest.raises(ValueError, match="nonexistent_geom"):
        MinkIK(
            mj.model, mj.data.qpos.copy(), dt=0.01,
            ee_site_name=r0.prefixed_ee_site,
            arm_joint_names=r0.prefixed_arm_joints,
            ik_config=ik_cfg,
        )


def test_camera_renderer(pick_place_env):
    sc, mj = pick_place_env
    mj.reset()
    for i, j in enumerate(sc.robots[0].prefixed_arm_joints):
        mj.set_joint_qpos(j, sc.robots[0].default_qpos[i])
    mj.forward()
    cr = CameraRenderer(sc.cameras, mj.model, mj.data)
    frames = cr.render_all(mj.data)
    # 当前场景：全局相机 + 手眼相机（2 个）
    assert len(frames) == 2
    for f in frames:
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
    assert "PushTTeacher" in TEACHER_REGISTRY
    assert TEACHER_CONFIG_REGISTRY["PickPlaceTeacher"].__name__ == "PickPlaceTeacherConfig"
    assert TEACHER_CONFIG_REGISTRY["PushTTeacher"].__name__ == "PushTTeacherConfig"
    # 配置自动加载（通过注册表而非硬编码）
    tc = load_teacher_config("configs/teachers/pick_place.yaml")
    assert tc.type == "PickPlaceTeacher"


def test_teacher_check_success_pick_place(pick_place_env):
    """评估成功判定由 teacher 提供：pick_place 在 cube 放到 plate 上时判成功。"""
    from lerobot_env_mujoco_lerobot import MujocoLerobotEnv

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
    from lerobot_env_mujoco_lerobot import MujocoLerobotEnv
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
    assert sc.view.lookat == (0.0, 0.0, 0.65)
    assert sc.view.distance > 0
    assert sc.view.image_size == (640, 480)
    sc_dual = load_scene_config("dual_pick_place", "configs/tasks/tasks.yaml")
    assert sc_dual.view.distance > 0
    assert sc_dual.view.image_size == (640, 480)


def test_env_render_view():
    """eval 视频用 view 配置的自由视角渲染，且不改变观测相机视角。"""
    from lerobot_env_mujoco_lerobot import MujocoLerobotEnv

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


def test_env_preprocessor_scales_rgb_and_depth():
    """评估 env preprocessor：rgb /255 → [0,1]；depth 默认米不变、mm 时 ×1000。"""
    import torch
    from lerobot_env_mujoco_lerobot.lerobot_env_cfg import (
        MujocoLerobotEnvConfig,
        ScaleDepthToOutputUnitProcessorStep,
        ScaleRgbImagesProcessorStep,
    )

    step_rgb = ScaleRgbImagesProcessorStep()
    out_rgb = step_rgb.observation(
        {"observation.images.cam.rgb": torch.full((3, 4, 4), 128.0)}
    )
    assert out_rgb["observation.images.cam.rgb"].tolist()[0][0][0] == pytest.approx(
        128.0 / 255.0, abs=1e-6
    )

    # depth 默认 "m"：保持不变（env 已是米，训练统一米）
    step_m = ScaleDepthToOutputUnitProcessorStep("m")
    out_m = step_m.observation(
        {"observation.images.cam.depth": torch.full((1, 4, 4), 0.7)}
    )
    assert out_m["observation.images.cam.depth"].tolist()[0][0][0] == pytest.approx(0.7)

    # depth "mm"：×1000
    step_mm = ScaleDepthToOutputUnitProcessorStep("mm")
    out_mm = step_mm.observation(
        {"observation.images.cam.depth": torch.full((1, 4, 4), 0.7)}
    )
    assert out_mm["observation.images.cam.depth"].tolist()[0][0][0] == 700.0

    # get_env_processors 默认含两个步骤，depth 默认 "m"
    env_pre, _ = MujocoLerobotEnvConfig(
        task="pick_place", dataset_config="configs/dataset/dataset_pick_place.yaml"
    ).get_env_processors()
    step_types = [type(s) for s in env_pre.steps]
    assert ScaleRgbImagesProcessorStep in step_types
    assert ScaleDepthToOutputUnitProcessorStep in step_types
    assert env_pre.steps[1].depth_output_unit == "m"


def test_eval_depth_normalization_matches_training():
    """训练与评估在统一深度单位下归一化一致（默认米，也可统一为 mm）。

    训练：dataloader 以 ``dataset.depth_output_unit``（本项目 "m"）解码深度、stats 为米；
    评估：env 产出米制 depth，env preprocessor 默认 "m" 不变 → Normalizer（米 stats）
    输入分布与训练一致。若训练/评估统一设为 "mm"，env ×1000 后得到相同归一化值。
    """
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_policy_config
    from lerobot_env_mujoco_lerobot.lerobot_env_cfg import MujocoLerobotEnvConfig
    from lerobot_policy_Adaptive_ACT.processor_adaptive_act import (
        make_adaptive_act_pre_post_processors,
    )

    def _build(unit: str, depth_mean: float, depth_std: float):
        cfg = make_policy_config(
            "adaptive_act",
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
                "observation.images.cam.rgb": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 32, 32)
                ),
                "observation.images.cam.depth": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(1, 32, 32)
                ),
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
            use_vae=False,
            chunk_size=8,
            n_action_steps=8,
            dim_model=32,
            n_heads=2,
            dim_feedforward=128,
            n_encoder_layers=1,
            n_decoder_layers=1,
            latent_dim=4,
            pretrained_backbone_weights=None,
        )
        stats = {
            "observation.state": {"mean": torch.zeros(7), "std": torch.ones(7)},
            "observation.images.cam.rgb": {
                "mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1) * 0.5,
            },
            "observation.images.cam.depth": {
                "mean": torch.full((1, 1, 1), depth_mean),
                "std": torch.full((1, 1, 1), depth_std),
            },
            "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
        }
        pre, _ = make_adaptive_act_pre_post_processors(cfg, dataset_stats=stats)
        env_pre, _ = MujocoLerobotEnvConfig(
            task="pick_place",
            dataset_config="configs/dataset/dataset_pick_place.yaml",
            depth_output_unit=unit,
        ).get_env_processors()
        return pre, env_pre

    def _normed_depth(pre, env_pre, depth_m: float):
        b = 2
        batch = {
            "observation.state": torch.randn(b, 7),
            "observation.images.cam.rgb": (torch.rand(b, 3, 32, 32) * 255).to(torch.uint8),
            "observation.images.cam.depth": torch.full((b, 1, 32, 32), depth_m),  # 米
            "action": torch.zeros(b, 8, 7),
            "action_is_pad": torch.zeros(b, 8, dtype=torch.bool),
            "task": ["t"] * b,
        }
        env_obs = env_pre(batch)
        normed = pre(env_obs)
        return env_obs["observation.images.cam.depth"], normed["observation.images.cam.depth"].cpu()

    # 统一米：训练 stats 米(0.311/0.201)，env 0.7m 不变 → (0.7-0.311)/0.201 ≈ 1.935
    pre_m, env_pre_m = _build("m", 0.311, 0.201)
    env_depth_m, normed_m = _normed_depth(pre_m, env_pre_m, 0.7)
    assert torch.allclose(env_depth_m, torch.full((2, 1, 32, 32), 0.7))
    assert torch.allclose(
        normed_m, torch.full((2, 1, 32, 32), (0.7 - 0.311) / 0.201), atol=1e-3
    )

    # 统一 mm：训练 stats 毫米(311/201)，env 0.7m ×1000=700mm → (700-311)/201 ≈ 1.935
    pre_mm, env_pre_mm = _build("mm", 311.0, 201.0)
    env_depth_mm, normed_mm = _normed_depth(pre_mm, env_pre_mm, 0.7)
    assert torch.allclose(env_depth_mm, torch.full((2, 1, 32, 32), 700.0))
    assert torch.allclose(
        normed_mm, torch.full((2, 1, 32, 32), (700.0 - 311.0) / 201.0), atol=1e-3
    )
    # 无论统一为 m 还是 mm，0.7m 深度的归一化结果一致
    assert torch.allclose(normed_m, normed_mm, atol=1e-3)


def test_depth_unit_explicitly_unified_as_m():
    """深度单位显式统一为米（不依赖补丁 / 自动识别 / 读取 meta）。

    记录侧：采集时在 features info 里显式写 depth_unit="m"（info.json 记录米，
    与 MuJoCo 米制渲染一致）。
    训练侧：显式 dataset.depth_output_unit=m（见 configs/policy/adaptive_act.yaml 的
    dataset.depth_output_unit），解码/stats 为米，与评估 env（米）一致。
    """
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot_policy_Adaptive_ACT import AdaptiveACTConfig  # noqa: F401 注册策略

    from mujoco_lerobot.data.dataset_writer import (
        LeRobotDatasetConfig,
        LeRobotDatasetWriter,
    )

    # ── 记录侧：features 的 depth info 显式声明米 ──
    class _FakeCam:
        name = "cam"
        height = 32
        width = 32

    feats = LeRobotDatasetWriter._default_features(
        LeRobotDatasetConfig(repo_id="x", root="/tmp/x", fps=30, cameras=[_FakeCam()])
    )
    assert feats["observation.images.cam.depth"]["info"] == {
        "is_depth_map": True, "depth_unit": "m",
    }

    # ── 训练侧：显式 depth_output_unit=m → 解码/stats 为米 ──
    cfg = draccus.parse(TrainPipelineConfig, args=[
        "--policy.type=adaptive_act",
        "--policy.input_features={\"observation.state\":{\"type\":\"STATE\",\"shape\":[7]}}",
        "--dataset.repo_id=mujoco_pick_place",
        "--dataset.root=outputs/datasets/pick_place/20260806_004855",
        "--dataset.depth_output_unit=m",
        "--output_dir=/tmp/depth_unit_explicit_test",
        "--steps=1",
    ])
    assert cfg.dataset.depth_output_unit == "m"

    ds, _ = make_train_eval_datasets(cfg)
    assert ds.depth_output_unit == "m"
    dep = np.asarray(ds[0]["observation.images.realsense_link_CAMERA.depth"]).astype(np.float32)
    assert 0.1 < float(dep.mean()) < 2.0  # 米量级（若是 mm 会 ~数百）


def test_env_reset_seed_controls_object_randomization():
    """env.reset(seed) 真正控制物体随机化（gym 语义）：同 seed 可复现、不同 seed 不同。"""
    from lerobot_env_mujoco_lerobot import MujocoLerobotEnv

    env = MujocoLerobotEnv(
        task_name="pick_place",
        dataset_config="configs/dataset/dataset_pick_place.yaml",
    )

    def cube_xy():
        mj = env._mj  # reset 后惰性初始化
        return mj.data.xpos[mj._body_ids["cube"]][:2].copy()

    env.reset(seed=1000)
    a = cube_xy()
    env.reset(seed=1000)
    b = cube_xy()
    assert np.allclose(a, b), "同 seed reset 应可复现"

    env.reset(seed=1001)
    c = cube_xy()
    assert not np.allclose(a, c), "不同 seed 应得到不同随机化"
    env.close()
