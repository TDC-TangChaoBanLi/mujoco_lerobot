"""Adaptive ACT 策略单元测试。

覆盖：
- 任意图像通道数输入（1 / 3 / 4 通道），非 3 通道仅替换 conv1，其余层保留预训练权重；
- 相机按分组共享 / 独立 resnet18 backbone（未指定相机自动归入默认组）；
- ``image_channels=None`` 自动从 input_features 识别通道数；
- 配置文件（YAML）与命令行参数优先级（CLI 覆盖 YAML）；
- 全输入输出归一化（pre/post processor MEAN_STD 数值 round-trip）；
- save/load、训练前向/反向、推理 select_action、无 VAE / 时序集成。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_policy_config

from lerobot_policy_Adaptive_ACT import AdaptiveACTConfig, AdaptiveACTPolicy
from lerobot_policy_Adaptive_ACT.configuration_adaptive_act import AdaptiveACTConfig as _Cfg
from lerobot_policy_Adaptive_ACT.processor_adaptive_act import (
    make_adaptive_act_pre_post_processors,
)

PRETRAINED = "ResNet18_Weights.IMAGENET1K_V1"
IMG = 32  # 测试用小分辨率


def _vis(c: int, h: int = IMG, w: int = IMG) -> PolicyFeature:
    return PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))


def _state(d: int = 7) -> PolicyFeature:
    return PolicyFeature(type=FeatureType.STATE, shape=(d,))


def _action(d: int = 7) -> PolicyFeature:
    return PolicyFeature(type=FeatureType.ACTION, shape=(d,))


def _make_cfg(**overrides) -> AdaptiveACTConfig:
    """构造一个小尺寸的 Adaptive ACT 配置（默认 3 相机、4 通道、自动识别）。"""
    base = dict(
        input_features={
            "observation.state": _state(),
            "observation.images.cam_left.rgb": _vis(4),
            "observation.images.cam_right.rgb": _vis(4),
            "observation.images.cam_top.rgb": _vis(4),
        },
        output_features={"action": _action()},
        camera_backbone_groups={"hand": ["cam_left", "cam_right"]},
        use_vae=True,
        chunk_size=8,
        n_action_steps=8,
        dim_model=64,
        n_heads=4,
        dim_feedforward=256,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        latent_dim=8,
        pretrained_backbone_weights=None,
    )
    base.update(overrides)
    return make_policy_config("adaptive_act", **base)


def _make_policy(**overrides) -> AdaptiveACTPolicy:
    return AdaptiveACTPolicy(_make_cfg(**overrides))


def _make_batch(cfg: AdaptiveACTConfig, b: int = 2) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key, ft in cfg.input_features.items():
        shape = tuple(ft.shape)
        batch[key] = torch.randn(b, *shape)
    batch["action"] = torch.randn(b, cfg.chunk_size, 7)
    batch["action_is_pad"] = torch.zeros(b, cfg.chunk_size, dtype=torch.bool)
    batch["task"] = ["task"] * b
    return batch


def _obs_batch(cfg: AdaptiveACTConfig, b: int = 1) -> dict[str, torch.Tensor]:
    obs: dict[str, torch.Tensor] = {}
    for key, ft in cfg.input_features.items():
        shape = tuple(ft.shape)
        obs[key] = torch.randn(b, *shape)
    return obs


# ── 注册与插件发现 ───────────────────────────────────────


def test_registered_and_discoverable():
    """注册为 adaptive_act，且按 lerobot 命名约定可解析出策略类。"""
    assert "adaptive_act" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("adaptive_act") is AdaptiveACTConfig
    assert get_policy_class("adaptive_act") is AdaptiveACTPolicy


def test_make_policy_config_roundtrip():
    """make_policy_config 能构造配置，且 encode/decode 保留 dict[str,list[str]] 字段。"""
    import json

    import draccus

    cfg = _make_cfg()
    enc = draccus.encode(cfg)
    assert enc["camera_backbone_groups"] == {"hand": ["cam_left", "cam_right"]}
    back = draccus.decode(AdaptiveACTConfig, enc)
    assert back.camera_backbone_groups == cfg.camera_backbone_groups
    assert back.type == "adaptive_act"


# ── 任意通道数 + 预训练保留 ───────────────────────────────


def test_conv1_adaptation_preserves_pretrained():
    """4 通道输入：conv1 重建为 4 通道，resnet 其余层保留 ImageNet 预训练权重。"""
    cfg = _make_cfg(pretrained_backbone_weights=PRETRAINED)
    policy = AdaptiveACTPolicy(cfg)
    backbone = policy.model.backbones["default"]  # IntermediateLayerGetter

    # conv1 通道自适应
    assert backbone.conv1.in_channels == 4
    assert backbone.conv1.out_channels == 64
    assert backbone.conv1.kernel_size == (7, 7)

    # 参考：相同结构、3 通道、预训练的 resnet18（FrozenBatchNorm2d）
    ref = torchvision.models.resnet18(
        weights=PRETRAINED, norm_layer=FrozenBatchNorm2d
    )
    ref_sd = {k: v.clone() for k, v in ref.state_dict().items()}

    our_sd = backbone.state_dict()
    conv1_key = "conv1.weight"
    # 非 conv1 层必须与预训练完全一致（IntermediateLayerGetter 顶层注册子模块，键无前缀；
    # 特征提取器只到 layer4，不含 fc）
    for key, ref_v in ref_sd.items():
        if key == "conv1.weight" or key == "conv1.bias" or key.startswith("fc."):
            continue
        assert key in our_sd, f"缺少预训练权重键 {key}"
        assert torch.equal(our_sd[key], ref_v), f"预训练权重被改动: {key}"

    # conv1：前 3 通道 = 预训练 RGB，多余通道 = RGB 均值
    ref_conv1 = ref_sd["conv1.weight"]  # (64, 3, 7, 7)
    our_conv1 = our_sd[conv1_key]  # (64, 4, 7, 7)
    assert torch.equal(our_conv1[:, :3], ref_conv1)
    expected_extra = ref_conv1.mean(dim=1, keepdim=True).expand(-1, 1, -1, -1)
    assert torch.allclose(our_conv1[:, 3:], expected_extra, atol=1e-6)


def test_rgb_channel3_keeps_pretrained_conv1():
    """3 通道输入：conv1 不重建，直接与预训练一致。"""
    cfg = _make_cfg(pretrained_backbone_weights=PRETRAINED)
    # 3 通道输入
    cfg.input_features = {
        k: (ft if "cam" not in k else PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG, IMG)))
        for k, ft in cfg.input_features.items()
    }
    policy = AdaptiveACTPolicy(cfg)
    backbone = policy.model.backbones["default"]
    ref = torchvision.models.resnet18(weights=PRETRAINED, norm_layer=FrozenBatchNorm2d)
    assert torch.equal(
        backbone.state_dict()["conv1.weight"], ref.state_dict()["conv1.weight"]
    )


def test_gray_channel1_forward():
    """1 通道（灰度）输入：conv1 为 1 通道，前向/反向正常。"""
    cfg = _make_cfg(
        input_features={
            "observation.state": _state(),
            "observation.images.cam_left.rgb": _vis(1),
            "observation.images.cam_right.rgb": _vis(1),
        },
        camera_backbone_groups=None,
        pretrained_backbone_weights=PRETRAINED,
    )
    policy = AdaptiveACTPolicy(cfg)
    assert policy.model.image_channels == 1
    assert policy.model.backbones["default"].conv1.in_channels == 1
    # 灰度 conv1 = 预训练 RGB 三通道均值
    ref = torchvision.models.resnet18(weights=PRETRAINED, norm_layer=FrozenBatchNorm2d)
    expected = ref.state_dict()["conv1.weight"].mean(dim=1, keepdim=True)
    assert torch.allclose(
        policy.model.backbones["default"].state_dict()["conv1.weight"], expected, atol=1e-6
    )

    policy.train()
    loss, _ = policy.forward(_make_batch(policy.config))
    loss.backward()
    assert torch.isfinite(loss)


def test_auto_channel_detection():
    """image_channels=None：从 input_features 自动识别为 4 通道。"""
    cfg = _make_cfg(image_channels=None)  # 特征为 4 通道
    policy = AdaptiveACTPolicy(cfg)
    assert policy.model.image_channels == 4


def test_image_channels_explicit_override():
    """显式 image_channels 与特征通道一致时可构建。"""
    cfg = _make_cfg(image_channels=4)
    policy = AdaptiveACTPolicy(cfg)
    assert policy.model.image_channels == 4


def test_mixed_channels_error():
    """不同相机通道数不一致时构建报错。"""
    cfg = _make_cfg(
        input_features={
            "observation.state": _state(),
            "observation.images.cam_left.rgb": _vis(3),
            "observation.images.cam_right.rgb": _vis(4),
        },
    )
    with pytest.raises(ValueError, match="same number of channels"):
        AdaptiveACTPolicy(cfg)


def test_image_channels_mismatch_error():
    """显式 image_channels 与特征通道不一致时 validate_features 报错。"""
    cfg = _make_cfg(image_channels=3)
    with pytest.raises(ValueError, match="does not match"):
        AdaptiveACTPolicy(cfg)


# ── 相机分组 backbone ─────────────────────────────────────


def test_backbone_group_routing():
    """未指定相机自动归入默认组，且各组为独立 resnet。"""
    cfg = _make_cfg(pretrained_backbone_weights=None)  # 随机初始化以便区分各组
    policy = AdaptiveACTPolicy(cfg)
    assert list(policy.model.backbones.keys()) == ["hand", "default"]
    assert policy.model.camera_backbone_map == {
        "cam_left": "hand",
        "cam_right": "hand",
        "cam_top": "default",
    }
    # 两个 backbone 是独立模块（随机初始化 → 权重不同）
    hand_w = policy.model.backbones["hand"].conv1.weight
    default_w = policy.model.backbones["default"].conv1.weight
    assert not torch.allclose(hand_w, default_w)
    # 同一输入经不同 backbone 输出不同特征
    img = torch.randn(1, 4, IMG, IMG)
    f_hand = policy.model.backbones["hand"](img)["feature_map"]
    f_default = policy.model.backbones["default"](img)["feature_map"]
    assert not torch.allclose(f_hand, f_default)


def test_all_cameras_unspecified_share_single_backbone():
    """未配置分组时所有相机共用一个 backbone（等价原生 ACT）。"""
    cfg = _make_cfg(camera_backbone_groups=None)
    policy = AdaptiveACTPolicy(cfg)
    assert list(policy.model.backbones.keys()) == ["default"]
    assert set(policy.model.camera_backbone_map.values()) == {"default"}
    assert len(policy.model.backbones) == 1


def test_all_cameras_specified_no_default_group():
    """所有相机都显式分组时不再创建默认组。"""
    cfg = _make_cfg(
        camera_backbone_groups={
            "hand": ["cam_left", "cam_right"],
            "global": ["cam_top"],
        },
    )
    policy = AdaptiveACTPolicy(cfg)
    assert list(policy.model.backbones.keys()) == ["hand", "global"]
    assert "default" not in policy.model.backbones


# ── 视觉特征通道拼接（RGBD） ───────────────────────────────


def _make_concat_cfg(n_cams: int = 1, **overrides) -> AdaptiveACTConfig:
    """构造 rgb+depth → rgbd 拼接配置（默认 1 相机，3+1=4 通道）。"""
    input_features = {"observation.state": _state()}
    concat = {}
    for i in range(n_cams):
        cam = "cam" if n_cams == 1 else f"cam{i}"
        input_features[f"observation.images.{cam}.rgb"] = _vis(3)
        input_features[f"observation.images.{cam}.depth"] = _vis(1)
        concat[f"observation.images.{cam}.rgbd"] = [
            f"observation.images.{cam}.rgb",
            f"observation.images.{cam}.depth",
        ]
    base = dict(
        input_features=input_features,
        output_features={"action": _action()},
        concat_visual_features=concat,
        use_vae=True,
        chunk_size=8,
        n_action_steps=8,
        dim_model=64,
        n_heads=4,
        dim_feedforward=256,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        latent_dim=8,
        pretrained_backbone_weights=None,
    )
    base.update(overrides)
    return make_policy_config("adaptive_act", **base)


def test_concat_rgbd_stacking():
    """rgb(3)+depth(1) 拼成 4 通道 RGBD：有效视图唯一、conv1=4 通道、前向正常。"""
    cfg = _make_concat_cfg()
    assert cfg.effective_visual_channels == {
        "observation.images.cam.rgbd": 4,
    }
    policy = AdaptiveACTPolicy(cfg)
    assert policy.model.image_channels == 4
    assert policy.model._effective_image_keys == [
        "observation.images.cam.rgbd"
    ]
    assert policy.model.backbones["default"].conv1.in_channels == 4

    policy.train()
    loss, _ = policy.forward(_make_batch(cfg))
    assert torch.isfinite(loss)
    loss.backward()

    policy.eval()
    policy.reset()
    action = policy.select_action(_obs_batch(cfg))
    assert tuple(action.shape) == (1, 7)


def test_concat_autodetect_and_explicit_channels():
    """image_channels=None 自动识别 4；显式 4 也可构建。"""
    cfg = _make_concat_cfg()
    policy = AdaptiveACTPolicy(cfg)
    assert policy.model.image_channels == 4
    cfg2 = _make_concat_cfg(image_channels=4)
    assert AdaptiveACTPolicy(cfg2).model.image_channels == 4


def test_concat_invalid_source_error():
    """concat 引用非 VISUAL 或不存在特征时报错。"""
    with pytest.raises(ValueError, match="VISUAL"):
        _make_concat_cfg(
            concat_visual_features={
                "observation.images.cam.rgbd": ["observation.state"]
            }
        )
    with pytest.raises(ValueError, match="VISUAL"):
        _make_concat_cfg(
            concat_visual_features={
                "observation.images.cam.rgbd": ["observation.images.nonexistent.rgb"]
            }
        )


def test_concat_view_key_conflict_error():
    """拼接视图键与 input_features 键冲突时报错。"""
    with pytest.raises(ValueError, match="冲突"):
        _make_concat_cfg(
            input_features={
                "observation.state": _state(),
                "observation.images.cam.rgb": _vis(3),
                "observation.images.cam.rgbd": _vis(4),  # 与拼接视图键冲突
            },
            concat_visual_features={
                "observation.images.cam.rgbd": ["observation.images.cam.rgb"],
            },
        )


def test_concat_multicam_mixed_with_plain():
    """多相机：部分相机 rgbd 拼接、部分相机纯 rgb，可混合且通道一致（4ch）。"""
    cfg = _make_cfg(
        input_features={
            "observation.state": _state(),
            "observation.images.cam_left.rgb": _vis(3),
            "observation.images.cam_left.depth": _vis(1),
            "observation.images.cam_right.rgbd": _vis(4),
        },
        concat_visual_features={
            "observation.images.cam_left.rgbd": [
                "observation.images.cam_left.rgb",
                "observation.images.cam_left.depth",
            ],
        },
        camera_backbone_groups=None,
    )
    assert set(cfg.effective_visual_channels.values()) == {4}
    policy = AdaptiveACTPolicy(cfg)
    assert policy.model.image_channels == 4
    assert len(policy.model._effective_image_keys) == 2
    policy.train()
    loss, _ = policy.forward(_make_batch(cfg))
    assert torch.isfinite(loss)


# ── 前向 / 反向 / 推理 ─────────────────────────────────────


def test_train_forward_backward():
    """训练前向（含 VAE）loss 有限，反向可用。"""
    cfg = _make_cfg()
    policy = AdaptiveACTPolicy(cfg)
    policy.train()
    loss, out = policy.forward(_make_batch(cfg))
    assert torch.isfinite(loss)
    assert set(out) == {"l1_loss", "kld_loss"}
    loss.backward()
    # 至少一个参数有梯度（含 backbone 分组）
    grads = [p.grad is not None for p in policy.parameters() if p.requires_grad]
    assert any(grads)


def test_no_vae_forward():
    """use_vae=False：推理式 latent（零向量）下训练仍可用。"""
    cfg = _make_cfg(use_vae=False)
    policy = AdaptiveACTPolicy(cfg)
    policy.train()
    loss, out = policy.forward(_make_batch(cfg))
    assert torch.isfinite(loss)
    assert "kld_loss" not in out
    loss.backward()


def test_select_action_shape_and_finite():
    """推理 select_action 返回 (B, action_dim) 且有限。"""
    cfg = _make_cfg()
    policy = AdaptiveACTPolicy(cfg)
    policy.eval()
    policy.reset()
    obs = _obs_batch(cfg)
    action = policy.select_action(obs)
    assert tuple(action.shape) == (1, 7)
    assert torch.isfinite(action).all()


def test_predict_action_chunk():
    """predict_action_chunk 返回 (B, chunk, action_dim)。"""
    cfg = _make_cfg()
    policy = AdaptiveACTPolicy(cfg)
    policy.eval()
    chunk = policy.predict_action_chunk(_obs_batch(cfg))
    assert tuple(chunk.shape) == (1, cfg.chunk_size, 7)


def test_temporal_ensemble():
    """时序集成：n_action_steps=1 时逐帧查询并输出动作。"""
    cfg = _make_cfg(
        temporal_ensemble_coeff=0.01,
        n_action_steps=1,
        chunk_size=8,
    )
    policy = AdaptiveACTPolicy(cfg)
    policy.eval()
    obs = _obs_batch(cfg)
    for _ in range(5):
        action = policy.select_action(obs)
        assert tuple(action.shape) == (1, 7)
        assert torch.isfinite(action).all()


# ── 配置优先级 ────────────────────────────────────────────


def test_config_file_and_cli_precedence(tmp_path):
    """YAML 配置文件 + CLI 参数：CLI 覆盖 YAML，未指定参数保留 YAML 值。"""
    import draccus

    yaml_path = tmp_path / "adaptive_act.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "chunk_size: 32",
                "n_action_steps: 32",
                "image_channels: 3",
                "use_vae: false",
                "camera_backbone_groups:",
                "  hand: [cam_left, cam_right]",
                "  global: [cam_top]",
            ]
        )
    )
    c1 = draccus.parse(AdaptiveACTConfig, config_path=str(yaml_path), args=[])
    assert c1.image_channels == 3
    assert c1.chunk_size == 32
    assert c1.use_vae is False
    assert c1.camera_backbone_groups == {"hand": ["cam_left", "cam_right"], "global": ["cam_top"]}

    # CLI 覆盖 YAML 中的同名参数；未覆盖的保留 YAML 值
    c2 = draccus.parse(
        AdaptiveACTConfig,
        config_path=str(yaml_path),
        args=["--image_channels=4", "--chunk_size=64"],
    )
    assert c2.image_channels == 4  # CLI 覆盖 3
    assert c2.chunk_size == 64  # CLI 覆盖 32
    assert c2.use_vae is False  # YAML 值保留
    assert c2.camera_backbone_groups == {"hand": ["cam_left", "cam_right"], "global": ["cam_top"]}


# ── 归一化 ────────────────────────────────────────────────


def _make_stats(cfg: AdaptiveACTConfig) -> dict:
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
    """pre/post processor：全部输入输出 MEAN_STD 归一化，且可反归一化还原。"""
    cfg = _make_cfg(use_vae=False, chunk_size=8, n_action_steps=8)
    stats = _make_stats(cfg)
    pre, post = make_adaptive_act_pre_post_processors(cfg, dataset_stats=stats)

    b = 2
    batch = {
        "observation.state": torch.ones(b, 7) * 10.0,
        "observation.images.cam_left.rgb": torch.ones(b, 4, IMG, IMG) * 4.0,
        "observation.images.cam_right.rgb": torch.ones(b, 4, IMG, IMG) * 4.0,
        "observation.images.cam_top.rgb": torch.ones(b, 4, IMG, IMG) * 4.0,
        "action": torch.ones(b, 8, 7) * 6.0,
        "action_is_pad": torch.zeros(b, 8, dtype=torch.bool),
        "task": ["t"] * b,
    }
    normed = pre(batch)
    # state: (10-0)/2 = 5
    assert torch.allclose(normed["observation.state"].cpu(), torch.ones(b, 7) * 5.0)
    # image: (4-0)/0.5 = 8
    for key in ["cam_left", "cam_right", "cam_top"]:
        assert torch.allclose(
            normed[f"observation.images.{key}.rgb"].cpu(), torch.ones(b, 4, IMG, IMG) * 8.0
        )
    # action: (6-0)/2 = 3
    assert torch.allclose(normed["action"].cpu(), torch.ones(b, 8, 7) * 3.0)

    # post 反归一化：3 * 2 = 6 还原
    unnormed = post(normed["action"])
    assert torch.allclose(unnormed.cpu(), torch.ones(b, 8, 7) * 6.0)


# ── 保存 / 加载 ───────────────────────────────────────────


def test_save_load_roundtrip(tmp_path):
    """save_pretrained / from_pretrained：config.json + model.safetensors，分组保留。"""
    cfg = _make_cfg(pretrained_backbone_weights=None)
    policy = AdaptiveACTPolicy(cfg)
    out = Path(tmp_path) / "ckpt"
    policy.save_pretrained(out, push_to_hub=False)
    saved = {p.name for p in out.iterdir()}
    assert "config.json" in saved
    assert "model.safetensors" in saved

    reloaded = AdaptiveACTPolicy.from_pretrained(out)
    assert isinstance(reloaded.config, AdaptiveACTConfig)
    assert reloaded.config.type == "adaptive_act"
    assert list(reloaded.model.backbones.keys()) == ["hand", "default"]
    # 权重一致
    a = policy.model.state_dict()
    b = reloaded.model.state_dict()
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k].cpu(), b[k].cpu()), f"权重不一致: {k}"
