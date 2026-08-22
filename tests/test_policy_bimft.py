"""BiMFT 策略单元测试。

覆盖：
- 注册与插件发现（"bimft" 类型）；
- 相机角色解析（显式 camera_roles / 自动检测）；
- ``_build_policy_batch`` 训练/推理形状（回归修复：多速率单帧 [B,R,D]
  不再被误当 [B,T,D]）；
- ``_reshape_target_to_chunk`` 动作块对齐；
- 小模型训练前向/反向；
- 推理历史队列（warmup 首帧补齐 + reset 清空）；
- pre/post processor 归一化数值往返；
- save/load roundtrip。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_policy_config

from lerobot_policy_bimft import BiMFTConfig, BiMFTPolicy
from lerobot_policy_bimft.modeling_bimft import (
    _build_policy_batch,
    _reshape_target_to_chunk,
    _resolve_camera_roles,
)
from lerobot_policy_bimft.processor_bimft import make_bimft_pre_post_processors

IMG = 32  # 测试用小分辨率
CAM_LEFT = "A_realsense_link_CAMERA"
CAM_RIGHT = "B_realsense_link_CAMERA"
CAM_GLOBAL = "global_realsense_link_CAMERA"
CAMERAS = [CAM_LEFT, CAM_RIGHT, CAM_GLOBAL]

# 小尺寸模型配置（覆盖包内 bimft.yaml 的 token/vision/action 等）
SMALL_MODEL_OVERRIDES = {
    "d_model": 64,
    "n_heads": 4,
    "dim_feedforward": 256,
    "grid_h": 4,
    "grid_w": 4,
    "pretrained_rgb": False,
    "action_decoder_layers": 1,
    "action_head_hidden_dim": 32,
    "latent_dim": 8,
    "cvae_encoder_layers": 1,
    "cvae_heads": 4,
    "joint_encoder_layers": 1,
    "force_encoder_layers": 1,
    "slot_fusion_layers": 1,
    "arm_temporal_layers": 1,
    "global_temporal_layers": 1,
    "coordination_layers": 1,
    "reinjection_layers": 1,
}


def _vis(c: int, h: int = IMG, w: int = IMG) -> PolicyFeature:
    return PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))


def _state(shape=(3, 14)) -> PolicyFeature:
    return PolicyFeature(type=FeatureType.STATE, shape=shape)


def _action(shape=(3, 14)) -> PolicyFeature:
    return PolicyFeature(type=FeatureType.ACTION, shape=shape)


def _input_features() -> dict[str, PolicyFeature]:
    feats = {
        "observation.state.joint.position": _state(),
        "observation.state.sensor.force": _state((3, 6)),
        "observation.state.sensor.torque": _state((3, 6)),
    }
    for cam in CAMERAS:
        feats[f"observation.images.{cam}.rgb"] = _vis(3)
        feats[f"observation.images.{cam}.depth"] = _vis(1)
    return feats


def _make_cfg(**overrides) -> BiMFTConfig:
    """构造小尺寸 BiMFT 配置（默认 3 相机、多速率 3 子采样）。"""
    base = dict(
        input_features=_input_features(),
        output_features={"action": _action()},
        camera_roles={
            "left": CAM_LEFT,
            "right": CAM_RIGHT,
            "global": CAM_GLOBAL,
        },
        n_obs_steps=4,
        horizon=30,
        n_action_steps=3,
        device="cpu",
    )
    base.update(overrides)
    return make_policy_config("bimft", **base)


def _make_policy(**overrides) -> BiMFTPolicy:
    cfg = _make_cfg(**overrides)
    # 用临时小尺寸 YAML 覆盖包内默认（避免 512 维大模型）
    import yaml

    yaml_path = Path(__file__).resolve().parents[1] / "user/lerobot_policy/lerobot_policy_bimft/lerobot_policy_bimft/bimft.yaml"
    data = yaml.safe_load(yaml_path.read_text())
    for section, key, val in [
        ("token", "d_model", SMALL_MODEL_OVERRIDES["d_model"]),
        ("token", "n_heads", SMALL_MODEL_OVERRIDES["n_heads"]),
        ("token", "dim_feedforward", SMALL_MODEL_OVERRIDES["dim_feedforward"]),
        ("vision", "grid_h", SMALL_MODEL_OVERRIDES["grid_h"]),
        ("vision", "grid_w", SMALL_MODEL_OVERRIDES["grid_w"]),
        ("vision", "pretrained_rgb", SMALL_MODEL_OVERRIDES["pretrained_rgb"]),
        ("action", "decoder_layers", SMALL_MODEL_OVERRIDES["action_decoder_layers"]),
        ("action", "head_hidden_dim", SMALL_MODEL_OVERRIDES["action_head_hidden_dim"]),
        ("cvae", "latent_dim", SMALL_MODEL_OVERRIDES["latent_dim"]),
        ("cvae", "encoder_layers", SMALL_MODEL_OVERRIDES["cvae_encoder_layers"]),
        ("cvae", "heads", SMALL_MODEL_OVERRIDES["cvae_heads"]),
        ("state", "joint_encoder_layers", SMALL_MODEL_OVERRIDES["joint_encoder_layers"]),
        ("state", "force_encoder_layers", SMALL_MODEL_OVERRIDES["force_encoder_layers"]),
        ("slot_fusion", "layers", SMALL_MODEL_OVERRIDES["slot_fusion_layers"]),
        ("temporal", "arm_layers", SMALL_MODEL_OVERRIDES["arm_temporal_layers"]),
        ("temporal", "global_layers", SMALL_MODEL_OVERRIDES["global_temporal_layers"]),
        ("bimanual", "coordination_layers", SMALL_MODEL_OVERRIDES["coordination_layers"]),
        ("bimanual", "reinjection_layers", SMALL_MODEL_OVERRIDES["reinjection_layers"]),
    ]:
        data[section][key] = val
    tmp = Path("/tmp/bimft_test_small.yaml")
    tmp.write_text(yaml.safe_dump(data))
    cfg.model_cfg_path = str(tmp)
    return BiMFTPolicy(cfg)


def _make_train_batch(cfg: BiMFTConfig, b: int = 2) -> dict[str, torch.Tensor]:
    """训练形状 batch（dataloader 输出：observation_delta_indices 堆叠 4 帧）。"""
    batch: dict[str, torch.Tensor] = {}
    for key, ft in cfg.input_features.items():
        shape = tuple(ft.shape)
        if ft.type is FeatureType.VISUAL:
            batch[key] = torch.randn(b, cfg.n_obs_steps, *shape)
        else:
            batch[key] = torch.randn(b, cfg.n_obs_steps, *shape)
    batch["action"] = torch.randn(b, cfg.horizon // 3, 3, 14)
    batch["action_is_pad"] = torch.zeros(b, cfg.horizon // 3, dtype=torch.bool)
    # LeRobot 生成的 observation *_is_pad mask
    batch["observation.state.joint.position_is_pad"] = torch.zeros(
        b, cfg.n_obs_steps, dtype=torch.bool
    )
    batch[f"observation.images.{CAM_LEFT}.rgb_is_pad"] = torch.zeros(
        b, cfg.n_obs_steps, dtype=torch.bool
    )
    batch["task"] = ["task"] * b
    return batch


def _make_infer_batch(cfg: BiMFTConfig, b: int = 1) -> dict[str, torch.Tensor]:
    """推理形状 batch（env 单帧多速率：state [B, R, D]、图像 [B, C, H, W]）。"""
    obs: dict[str, torch.Tensor] = {}
    for key, ft in cfg.input_features.items():
        shape = tuple(ft.shape)
        if ft.type is FeatureType.VISUAL:
            obs[key] = torch.randn(b, *shape)
        else:
            obs[key] = torch.randn(b, *shape)
    return obs


# ── 注册与插件发现 ───────────────────────────────────────


def test_registered_and_discoverable():
    """注册为 bimft，且按 lerobot 命名约定可解析出策略类。"""
    assert "bimft" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("bimft") is BiMFTConfig
    assert get_policy_class("bimft") is BiMFTPolicy


def test_make_policy_config_roundtrip():
    """make_policy_config 能构造配置，且 encode/decode 保留 camera_roles 字段。"""
    import json

    import draccus

    cfg = _make_cfg()
    enc = draccus.encode(cfg)
    assert enc["camera_roles"] == {
        "left": CAM_LEFT,
        "right": CAM_RIGHT,
        "global": CAM_GLOBAL,
    }
    back = draccus.decode(BiMFTConfig, enc)
    assert back.camera_roles == cfg.camera_roles
    assert back.type == "bimft"


def test_camera_roles_validation():
    """camera_roles 缺少角色时报错。"""
    with pytest.raises(ValueError, match="left/right/global"):
        _make_cfg(camera_roles={"left": CAM_LEFT, "right": CAM_RIGHT})


# ── 相机角色解析 ──────────────────────────────────────────


def test_resolve_camera_roles_explicit():
    """显式 camera_roles 优先。"""
    cfg = _make_cfg()
    roles = _resolve_camera_roles(cfg, cfg.input_features)
    assert roles == {"left": CAM_LEFT, "right": CAM_RIGHT, "global": CAM_GLOBAL}


def test_resolve_camera_roles_auto():
    """自动检测：global/A_/B_ 模式匹配。"""
    cfg = _make_cfg(camera_roles=None)
    roles = _resolve_camera_roles(cfg, cfg.input_features)
    assert roles == {"left": CAM_LEFT, "right": CAM_RIGHT, "global": CAM_GLOBAL}


def test_resolve_camera_roles_auto_insufficient():
    """相机不足 3 路时报错。"""
    cfg = _make_cfg(camera_roles=None)
    feats = {
        "observation.state.joint.position": _state(),
        "observation.images.cam1.rgb": _vis(3),
        "observation.images.cam2.rgb": _vis(3),
    }
    with pytest.raises(ValueError, match="left/right/global"):
        _resolve_camera_roles(cfg, feats)


# ── _build_policy_batch 形状 ──────────────────────────────


def test_build_policy_batch_train_shapes():
    """训练形状 [B,4,3,14] state + [B,4,3,480,640] 图像 → PolicyBatch 各键形状。"""
    cfg = _make_cfg()
    batch = _make_train_batch(cfg, b=2)
    pb = _build_policy_batch(batch, cfg)

    assert tuple(pb["left_wrist_rgbd"].shape) == (2, 4, 4, IMG, IMG)
    assert tuple(pb["right_wrist_rgbd"].shape) == (2, 4, 4, IMG, IMG)
    assert tuple(pb["global_rgbd"].shape) == (2, 4, 4, IMG, IMG)
    assert tuple(pb["left_joint_gripper"].shape) == (2, 4, 3, 7)
    assert tuple(pb["right_joint_gripper"].shape) == (2, 4, 3, 7)
    assert tuple(pb["left_wrench"].shape) == (2, 4, 3, 6)
    assert tuple(pb["right_wrench"].shape) == (2, 4, 3, 6)
    assert tuple(pb["image_time_offsets"].shape) == (2, 4)
    assert tuple(pb["high_rate_time_offsets"].shape) == (2, 4, 3)
    assert tuple(pb["image_valid_mask"].shape) == (2, 4)
    assert tuple(pb["high_rate_valid_mask"].shape) == (2, 4, 3)


def test_build_policy_batch_infer_shapes():
    """推理形状 [B,3,14] state + [B,3,480,640] 图像 → 正确 [B,4,3,7]。

    回归修复：原版把 [B,3,14] 的 3 个子采样误当时间维。
    """
    cfg = _make_cfg()
    batch = _make_infer_batch(cfg, b=1)
    pb = _build_policy_batch(batch, cfg)

    assert tuple(pb["left_wrist_rgbd"].shape) == (1, 4, 4, IMG, IMG)
    assert tuple(pb["left_joint_gripper"].shape) == (1, 4, 3, 7)
    assert tuple(pb["right_joint_gripper"].shape) == (1, 4, 3, 7)
    assert tuple(pb["left_wrench"].shape) == (1, 4, 3, 6)
    assert tuple(pb["right_wrench"].shape) == (1, 4, 3, 6)


def test_build_policy_batch_flat_state():
    """扁平单帧单采样 [B, D] state → 重复到 [B, T, R, D]。"""
    cfg = _make_cfg()
    batch = {
        "observation.state.joint.position": torch.randn(2, 14),
        "observation.state.sensor.force": torch.randn(2, 6),
        "observation.state.sensor.torque": torch.randn(2, 6),
        **{
            f"observation.images.{cam}.rgb": torch.randn(2, 3, IMG, IMG)
            for cam in CAMERAS
        },
        **{
            f"observation.images.{cam}.depth": torch.randn(2, 1, IMG, IMG)
            for cam in CAMERAS
        },
    }
    pb = _build_policy_batch(batch, cfg)
    assert tuple(pb["left_joint_gripper"].shape) == (2, 4, 3, 7)
    assert tuple(pb["left_wrench"].shape) == (2, 4, 3, 6)


def test_build_policy_batch_missing_state_error():
    """缺少 state key 时报错。"""
    cfg = _make_cfg()
    with pytest.raises(ValueError, match="state"):
        _build_policy_batch({"action": torch.randn(2, 30, 14)}, cfg)


# ── _reshape_target_to_chunk ──────────────────────────────


def test_reshape_target_to_chunk():
    """[B,10,3,14] → [B,30,14]；不足时复制最后一帧填充。"""
    target = torch.randn(2, 10, 3, 14)
    out = _reshape_target_to_chunk(target, 30)
    assert tuple(out.shape) == (2, 30, 14)
    # 前 30 步与 reshape 一致
    assert torch.equal(out, target.reshape(2, 30, 14))

    # 不足 K：复制最后一帧
    short = torch.randn(2, 5, 3, 14)
    out2 = _reshape_target_to_chunk(short, 30)
    assert tuple(out2.shape) == (2, 30, 14)
    assert torch.equal(out2[:, 15:], out2[:, 14:15].expand(-1, 15, -1))


# ── 前向 / 反向 ──────────────────────────────────────────


def test_train_forward_backward():
    """训练前向 loss 有限，反向可用。"""
    policy = _make_policy()
    policy.train()
    batch = _make_train_batch(policy.config, b=2)
    loss, out = policy.forward(batch)
    assert torch.isfinite(loss)
    assert set(out) == {
        "action_loss", "smooth_loss", "kl_loss", "left_loss", "right_loss",
    }
    loss.backward()
    grads = [p.grad is not None for p in policy.parameters() if p.requires_grad]
    assert any(grads)


def test_forward_missing_action():
    """batch 无 action 时返回零损失（不崩溃）。"""
    policy = _make_policy()
    policy.train()
    batch = _make_train_batch(policy.config, b=2)
    batch.pop("action")
    loss, out = policy.forward(batch)
    assert torch.isfinite(loss)


# ── 推理历史队列 ─────────────────────────────────────────


def test_select_action_history_queue():
    """历史队列：warmup 首帧补齐，连续调用输出 (B, 14)。"""
    policy = _make_policy()
    policy.eval()
    policy.reset()
    assert len(policy._obs_queue) == 0

    obs = _make_infer_batch(policy.config, b=1)
    for i in range(5):
        action = policy.select_action(obs)
        assert tuple(action.shape) == (1, 14)
        assert torch.isfinite(action).all()
        # 队列最多 n_obs_steps 帧
        assert len(policy._obs_queue) <= policy.config.n_obs_steps

    # 队列已满（4 帧）
    assert len(policy._obs_queue) == policy.config.n_obs_steps


def test_select_action_reset_clears_queue():
    """reset() 清空历史队列。"""
    policy = _make_policy()
    policy.eval()
    obs = _make_infer_batch(policy.config, b=1)
    policy.select_action(obs)
    # select_action 内部已把队列补齐到 n_obs_steps
    assert len(policy._obs_queue) == policy.config.n_obs_steps
    policy.reset()
    assert len(policy._obs_queue) == 0


def test_predict_action_chunk():
    """predict_action_chunk 返回 (B, horizon, 14)。"""
    policy = _make_policy()
    policy.eval()
    policy.reset()
    obs = _make_infer_batch(policy.config, b=1)
    chunk = policy.predict_action_chunk(obs)
    assert tuple(chunk.shape) == (1, policy.config.horizon, 14)
    assert torch.isfinite(chunk).all()


# ── 归一化 ────────────────────────────────────────────────


def _make_stats(cfg: BiMFTConfig) -> dict:
    stats = {}
    for key, ft in {**cfg.input_features, **cfg.output_features}.items():
        shape = tuple(ft.shape)
        if ft.type is FeatureType.VISUAL:
            mean = torch.full((shape[0], 1, 1), 0.0)
            std = torch.full((shape[0], 1, 1), 0.5)
        else:
            mean = torch.zeros(shape)
            std = torch.ones(shape) * 2.0
        stats[key] = {"mean": mean, "std": std}
    return stats


def test_normalization_roundtrip():
    """pre/post processor：STATE/ACTION MEAN_STD 归一化，VISUAL IDENTITY，可还原。"""
    cfg = _make_cfg()
    stats = _make_stats(cfg)
    pre, post = make_bimft_pre_post_processors(cfg, dataset_stats=stats)

    b = 2
    batch = {
        "observation.state.joint.position": torch.ones(b, 3, 14) * 10.0,
        "observation.state.sensor.force": torch.ones(b, 3, 6) * 10.0,
        "observation.state.sensor.torque": torch.ones(b, 3, 6) * 10.0,
        **{
            f"observation.images.{cam}.rgb": torch.ones(b, 3, IMG, IMG) * 4.0
            for cam in CAMERAS
        },
        **{
            f"observation.images.{cam}.depth": torch.ones(b, 1, IMG, IMG) * 4.0
            for cam in CAMERAS
        },
        "action": torch.ones(b, 3, 14) * 6.0,
        "action_is_pad": torch.zeros(b, 3, dtype=torch.bool),
        "task": ["t"] * b,
    }
    normed = pre(batch)
    # state: (10-0)/2 = 5
    assert torch.allclose(
        normed["observation.state.joint.position"].cpu(), torch.ones(b, 3, 14) * 5.0
    )
    # image: VISUAL IDENTITY → 不变
    for cam in CAMERAS:
        assert torch.allclose(
            normed[f"observation.images.{cam}.rgb"].cpu(), torch.ones(b, 3, IMG, IMG) * 4.0
        )
    # action: (6-0)/2 = 3
    assert torch.allclose(normed["action"].cpu(), torch.ones(b, 3, 14) * 3.0)

    # post 反归一化：3 * 2 = 6 还原
    unnormed = post(normed["action"])
    assert torch.allclose(unnormed.cpu(), torch.ones(b, 3, 14) * 6.0)


# ── 保存 / 加载 ───────────────────────────────────────────


def test_save_load_roundtrip(tmp_path):
    """save_pretrained / from_pretrained：config.json + model.safetensors + processor。"""
    policy = _make_policy()
    out = Path(tmp_path) / "ckpt"
    policy.save_pretrained(out, push_to_hub=False)
    saved = {p.name for p in out.iterdir()}
    assert "config.json" in saved
    assert "model.safetensors" in saved
    assert "policy_preprocessor.json" in saved
    assert "policy_postprocessor.json" in saved

    reloaded = BiMFTPolicy.from_pretrained(out)
    assert isinstance(reloaded.config, BiMFTConfig)
    assert reloaded.config.type == "bimft"
    # 权重一致
    a = policy.model.state_dict()
    b = reloaded.model.state_dict()
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k].cpu(), b[k].cpu()), f"权重不一致: {k}"

    # 加载后推理可用
    reloaded.eval()
    reloaded.reset()
    obs = _make_infer_batch(reloaded.config, b=1)
    action = reloaded.select_action(obs)
    assert tuple(action.shape) == (1, 14)


def test_get_optim_params():
    """优化器参数分组：backbone 小学习率 + 其余默认。"""
    policy = _make_policy()
    groups = policy.get_optim_params()
    assert len(groups) == 2
    assert groups[0]["lr"] == policy.config.optimizer_lr * 0.1
    assert "lr" not in groups[1] or groups[1]["lr"] == policy.config.optimizer_lr
    # 两组参数不相交且覆盖全部可训练参数
    p0 = {id(p) for p in groups[0]["params"]}
    p1 = {id(p) for p in groups[1]["params"]}
    assert not (p0 & p1)
    all_trainable = {id(p) for p in policy.parameters() if p.requires_grad}
    assert p0 | p1 == all_trainable