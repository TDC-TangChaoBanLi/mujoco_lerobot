"""AdaptiveACTConfig — 自适应 ACT 策略配置。

在原生 ACT 配置基础上新增两个字段：
- ``camera_backbone_groups``：dict[组名, list[相机短名]]，控制哪些相机共用一个
  resnet18 backbone；未出现在任何组的相机自动归入共享默认组。
- ``image_channels``：输入图像通道数；``None`` 表示从 ``input_features`` 自动识别。

其余超参与原生 ACT 一致，全部可经 YAML 配置文件或命令行传入
（命令行参数优先级更高，覆盖配置文件中的同名参数）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("adaptive_act")
@dataclass
class AdaptiveACTConfig(PreTrainedConfig):
    """Configuration class for the Adaptive Action Chunking Transformers policy.

    与原生 ACT 的主要差异（详见 modeling_adaptive_act.py）：
    - 视觉 backbone 支持任意输入通道数（灰度 / RGB / RGB-D / 自定义通道），
      非 3 通道时仅替换 resnet 的 conv1，其余层保留预训练权重。
    - 每个相机可指定所属的 backbone 分组（``camera_backbone_groups``），
      同组相机共享一个 resnet18，不同组使用独立 resnet18。
    - ``image_channels=None`` 时在模型构建阶段从 ``input_features`` 自动识别
      视觉特征的通道数，无需人工指定。

    Args 与原生 ACT 保持一致的部分不再重复说明：
        n_obs_steps, chunk_size, n_action_steps, normalization_mapping,
        vision_backbone, pretrained_backbone_weights,
        replace_final_stride_with_dilation, pre_norm, dim_model, n_heads,
        dim_feedforward, feedforward_activation, n_encoder_layers,
        n_decoder_layers, use_vae, latent_dim, n_vae_encoder_layers,
        temporal_ensemble_coeff, dropout, kl_weight, optimizer_*。
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ── Adaptive 新增字段 ────────────────────────────────
    # 每个视觉 backbone 分组的相机短名列表，如 {"hand": ["cam_left", "cam_right"]}。
    # 未指定的相机自动归入共享默认组（"default"）。None = 所有相机共用一个 backbone。
    camera_backbone_groups: dict[str, list[str]] | None = None
    # 输入图像通道数；None = 构建时从 input_features 的 VISUAL 特征 shape 自动识别。
    image_channels: int | None = None
    # 视觉特征通道拼接：把多个 VISUAL 输入特征沿通道维拼成一个视图（如 rgb+depth → rgbd）。
    # 键 = 拼接视图的 feature key（建议 observation.images.<cam>.rgbd），值 = 源特征 key 列表。
    # 视图通道数 = 各源通道数之和；视图所属相机由键中的相机短名决定（同 camera_backbone_groups 规则）。
    # 被拼接消费的源特征不再单独送入 backbone。
    concat_visual_features: dict[str, list[str]] | None = None

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.feedforward_activation not in ("relu", "gelu"):
            raise ValueError(
                f"`feedforward_activation` must be one of 'relu' or 'gelu' for "
                f"torch.nn.Transformer. Got {self.feedforward_activation!r}."
            )
        if self.image_channels is not None and self.image_channels < 1:
            raise ValueError(
                f"`image_channels` must be a positive integer or None (auto). Got {self.image_channels}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

        concat = self.concat_visual_features or {}
        for view, sources in concat.items():
            if not sources:
                raise ValueError(
                    f"`concat_visual_features[{view!r}]` must list at least one source feature."
                )
            if view in (self.input_features or {}):
                raise ValueError(
                    f"`concat_visual_features` 视图键 {view!r} 与 input_features 中的键冲突。"
                )
            for src in sources:
                ft = (self.input_features or {}).get(src)
                if ft is None or ft.type is not FeatureType.VISUAL:
                    raise ValueError(
                        f"`concat_visual_features[{view!r}]` 引用的源特征 {src!r} 必须是 "
                        f"input_features 中的 VISUAL 特征。"
                    )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )
        if self.image_features:
            channels = set(self.effective_visual_channels.values())
            if len(channels) > 1:
                raise ValueError(
                    f"All visual features must have the same number of channels. Got {channels}."
                )
            if self.image_channels is not None:
                inferred = next(iter(channels))
                if self.image_channels != inferred:
                    raise ValueError(
                        f"`image_channels`={self.image_channels} does not match the visual feature "
                        f"channels {inferred} inferred from `input_features`."
                    )

    @property
    def effective_image_keys(self) -> list[str]:
        """有序的有效视觉视图键：未被拼接消费的原始特征 + 拼接视图（按配置顺序）。"""
        concat = self.concat_visual_features or {}
        consumed: set[str] = set()
        for sources in concat.values():
            consumed.update(sources)
        raw = [k for k in self.image_features if k not in consumed]
        return raw + list(concat.keys())

    @property
    def effective_visual_channels(self) -> dict[str, int]:
        """每个有效视觉视图的通道数（拼接视图 = 各源通道数之和）。"""
        concat = self.concat_visual_features or {}
        out: dict[str, int] = {}
        for key in self.effective_image_keys:
            if key in concat:
                out[key] = sum(self.image_features[src].shape[0] for src in concat[key])
            else:
                out[key] = self.image_features[key].shape[0]
        return out

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
