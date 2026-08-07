"""AdaptiveACTPolicy — 自适应 ACT 策略。

在原生 ACT（Action Chunking Transformer）基础上扩展：
1. 任意图像通道数输入：视觉 backbone 的 conv1 按输入通道数自适应重建
   （C==3 保留预训练；C<3 取预训练 RGB 子集均值；C>3 前 3 通道预训练 + 多余通道
   取 RGB 均值），resnet 其余层全部保留预训练权重。
2. 相机按分组共享/独立 backbone：``camera_backbone_groups`` 指定各分组包含的
   相机，同组相机共用一个 resnet18；未指定的相机自动归入共享默认组。
3. ``image_channels=None`` 时从 ``input_features`` 的 VISUAL 特征自动识别通道数。
4. Transformer 全部基于 ``torch.nn.TransformerEncoder / TransformerDecoder`` 构建
   （不再依赖 lerobot 内部的 ACT 自定义层）。
5. 归一化由 LeRobot 默认 pre/post processor（MEAN_STD）完成。

结构与原生 ACT 一致：可选 VAE encoder（训练时）→ transformer encoder（输入 token
为 [latent, robot_state, env_state, *各相机特征图像素]）→ transformer decoder
（可学习 chunk 查询）→ action_head。
"""

from __future__ import annotations

import math
from collections import deque
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .configuration_adaptive_act import AdaptiveACTConfig


# ── 通用工具 ─────────────────────────────────────────────


def _camera_name_from_key(key: str) -> str:
    """从 feature key 提取相机短名。

    ``observation.images.<cam>.<suffix>`` → ``<cam>``；无法解析时原样返回。
    """
    parts = key.split(".")
    if len(parts) >= 3 and parts[:2] == ["observation", "images"]:
        return parts[2]
    return key


def _resolve_image_channels(config: AdaptiveACTConfig) -> int:
    """解析输入图像通道数：显式指定优先，否则从有效视觉视图（含拼接）自动识别。"""
    if config.image_channels is not None:
        return config.image_channels
    channels = set(config.effective_visual_channels.values())
    if not channels:
        return 3  # 无图像输入时该值不参与任何计算
    if len(channels) > 1:
        raise ValueError(
            f"All visual features must have the same number of channels. Got {channels}."
        )
    return channels.pop()


def _build_camera_backbone_map(config: AdaptiveACTConfig) -> dict[str, str]:
    """建立 相机短名 → backbone 分组名 的映射。

    显式分组的相机归入对应组；未指定的相机归入共享默认组 ``"default"``。
    """
    groups = config.camera_backbone_groups or {}
    mapping: dict[str, str] = {}
    specified: set[str] = set()
    for group, cams in groups.items():
        for cam in cams:
            mapping[cam] = group
            specified.add(cam)
    for key in config.effective_image_keys:
        cam = _camera_name_from_key(key)
        if cam not in specified:
            mapping[cam] = "default"
    return mapping


def _backbone_group_keys(config: AdaptiveACTConfig) -> list[str]:
    """backbone 分组的确定性顺序：显式分组（按 dict 顺序）+ 默认组（如有）。"""
    explicit = list((config.camera_backbone_groups or {}).keys())
    mapping = _build_camera_backbone_map(config)
    rest = [g for g in dict.fromkeys(mapping.values()) if g not in explicit]
    return explicit + rest


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D 正弦位置编码（Attention Is All You Need 风格）。"""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class SinusoidalPositionEmbedding2d(nn.Module):
    """2D 正弦位置编码（DETR / ACT 风格）。

    PyTorch 未提供现成的 2D 正弦位置编码模块，故在此实现一个小 helper。
    """

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )
        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)
        return pos_embed


class TemporalEnsembler:
    """ACT 的在线时序集成（Algorithm 2, https://huggingface.co/papers/2304.13705）。

    独立实现（非模块），逻辑与原生 ACT 一致。
    """

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self) -> None:
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


# ── Backbone（通道自适应 + 分组）─────────────────────────


def _adapt_conv1_channels(model: nn.Module, in_channels: int) -> nn.Module:
    """按输入通道数重建 resnet 的 conv1，其余层（含 BN）保持预训练权重。

    - ``in_channels == 3``：直接复用预训练 conv1。
    - ``in_channels < 3``：取预训练 RGB 全通道均值（标准灰度初始化，保留幅度）。
    - ``in_channels > 3``：前 3 通道拷贝预训练权重，多余通道用 RGB 通道均值初始化。
    """
    conv1 = model.conv1
    if conv1.in_channels == in_channels:
        return model
    new_conv1 = nn.Conv2d(
        in_channels, conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None,
    )
    with torch.no_grad():
        w = conv1.weight  # (out, 3, k, k)
        if in_channels < 3:
            mean_w = w.mean(dim=1, keepdim=True)
            new_conv1.weight.copy_(mean_w.expand(-1, in_channels, -1, -1))
        else:
            new_conv1.weight[:, :3] = w
            if in_channels > 3:
                extra = w.mean(dim=1, keepdim=True)
                new_conv1.weight[:, 3:] = extra.expand(-1, in_channels - 3, -1, -1)
        if conv1.bias is not None:
            new_conv1.bias.copy_(conv1.bias)
    model.conv1 = new_conv1
    return model


def _build_backbone(config: AdaptiveACTConfig, in_channels: int) -> tuple[nn.Module, int]:
    """构建一个带预训练权重的 resnet backbone，conv1 按通道数自适应。"""
    backbone_model = getattr(torchvision.models, config.vision_backbone)(
        replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
        weights=config.pretrained_backbone_weights,
        norm_layer=FrozenBatchNorm2d,
    )
    out_features = backbone_model.fc.in_features
    _adapt_conv1_channels(backbone_model, in_channels)
    backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
    return backbone, out_features


# ── 模型 ────────────────────────────────────────────────


class AdaptiveACT(nn.Module):
    """自适应 ACT 网络：可选 VAE encoder + 分组视觉 backbone + Transformer encoder/decoder。

    前向输入 ``batch`` 结构（与原生 ACT 一致）：
        [OBS_STATE]           (B, state_dim)          可选
        [OBS_ENV_STATE]       (B, env_dim)            可选
        OBS_IMAGES            list[(B, C, H, W)]      至少一个图像或 env_state
        ACTION                (B, chunk, action_dim)  仅训练 + use_vae 时需要
        action_is_pad         (B, chunk)              仅训练 + use_vae 时需要
    """

    def __init__(self, config: AdaptiveACTConfig):
        super().__init__()
        self.config = config

        self.image_channels = _resolve_image_channels(config)
        self.camera_backbone_map = _build_camera_backbone_map(config)
        # 视觉拼接：有效视图键（未拼接原始特征 + 拼接视图）、源特征、原始特征 key→索引
        self._effective_image_keys = config.effective_image_keys
        self._concat_sources = config.concat_visual_features or {}
        self._raw_key_to_index = {
            key: i for i, key in enumerate(config.image_features)
        }

        # ── VAE encoder（训练时把 [cls, robot_state, *action_sequence] 编码为 latent）──
        if config.use_vae:
            self.vae_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.dim_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    activation=config.feedforward_activation,
                    batch_first=False,
                    norm_first=config.pre_norm,
                ),
                num_layers=config.n_vae_encoder_layers,
                norm=nn.LayerNorm(config.dim_model) if config.pre_norm else None,
                enable_nested_tensor=False,
            )
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                config.action_feature.shape[0], config.dim_model
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # ── 分组视觉 backbone（相机→分组→对应 resnet）──
        self._backbone_out_features = 0
        if config.image_features:
            backbones: dict[str, nn.Module] = {}
            for group in _backbone_group_keys(config):
                backbone, out_features = _build_backbone(config, self.image_channels)
                self._backbone_out_features = out_features
                backbones[group] = backbone
            self.backbones = nn.ModuleDict(backbones)

        # ── Transformer encoder / decoder ──
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.dim_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.feedforward_activation,
                batch_first=False,
                norm_first=config.pre_norm,
            ),
            num_layers=config.n_encoder_layers,
            norm=nn.LayerNorm(config.dim_model) if config.pre_norm else None,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.dim_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.feedforward_activation,
                batch_first=False,
                norm_first=config.pre_norm,
            ),
            num_layers=config.n_decoder_layers,
            norm=nn.LayerNorm(config.dim_model),
        )

        # ── Transformer encoder 输入投影 ──
        if config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )
        if config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                self._backbone_out_features, config.dim_model, kernel_size=1
            )

        # ── 位置编码 ──
        n_1d_tokens = 1  # latent
        if config.robot_state_feature:
            n_1d_tokens += 1
        if config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if config.image_features:
            self.encoder_cam_feat_pos_embed = SinusoidalPositionEmbedding2d(config.dim_model // 2)

        # ── Transformer decoder 查询与输出头 ──
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, config.action_feature.shape[0])

    # ── 前向 ────────────────────────────────────────────

    def forward(
        self, batch: dict[str, Tensor],
    ) -> tuple[Tensor, tuple[Tensor | None, Tensor | None]]:
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = (
            batch[OBS_IMAGES][0].shape[0]
            if OBS_IMAGES in batch
            else batch[OBS_ENV_STATE].shape[0]
        )
        ref = batch.get(OBS_STATE, batch.get(OBS_ENV_STATE, batch[OBS_IMAGES][0]))
        device, dtype = ref.device, ref.dtype

        # ── latent（训练时经 VAE encoder，推理时为 0）──
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+1|2, D)

            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)

            # 位置编码加到输入上（nn.Transformer 不内置 pos 加法）
            vae_in = vae_encoder_input.permute(1, 0, 2) + pos_embed.permute(1, 0, 2)
            cls_token_out = self.vae_encoder(
                vae_in, src_key_padding_mask=key_padding_mask
            )[0]  # (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim], dtype=dtype, device=device
            )

        # ── 组装 transformer encoder 输入 token ──
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            raw_images = batch[OBS_IMAGES]  # 与 config.image_features 顺序一致（策略侧构建）
            # 逐个有效视图（未拼接的原始特征 + 通道拼接视图）按其所属分组使用对应 backbone
            for view_key in self._effective_image_keys:
                if view_key in self._concat_sources:
                    img = torch.cat(
                        [
                            raw_images[self._raw_key_to_index[src]]
                            for src in self._concat_sources[view_key]
                        ],
                        dim=1,
                    )
                else:
                    img = raw_images[self._raw_key_to_index[view_key]]
                cam = _camera_name_from_key(view_key)
                backbone = self.backbones[self.camera_backbone_map[cam]]
                cam_features = backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens + encoder_in_pos_embed)

        # ── Transformer decoder（可学习 chunk 查询 + 交叉注意力）──
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_out.dtype,
            device=encoder_out.device,
        )
        decoder_in = decoder_in + self.decoder_pos_embed.weight.unsqueeze(1)
        decoder_out = self.decoder(decoder_in, encoder_out)
        decoder_out = decoder_out.transpose(0, 1)  # (B, S, D)

        actions = self.action_head(decoder_out)
        return actions, (mu, log_sigma_x2)


# ── 策略 ────────────────────────────────────────────────


class AdaptiveACTPolicy(PreTrainedPolicy):
    """自适应 ACT 策略（LeRobot 插件，类型名 ``adaptive_act``）。"""

    config_class = AdaptiveACTConfig
    name = "adaptive_act"

    def __init__(
        self,
        config: AdaptiveACTConfig,
        **kwargs,
    ) -> None:
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = AdaptiveACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = TemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )

        self.reset()

    def get_optim_params(self) -> dict:
        # 视觉 backbone 使用独立的更小学习率（与原生 ACT 一致）
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbones") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbones") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self) -> None:
        """环境 reset 时调用。"""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """给定观测输出单步动作，内部用动作队列缓存 chunk。"""
        self.eval()  # 队列消费期间保持 eval 模式

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """给定观测预测一整个动作 chunk。"""
        self.eval()
        if self.config.image_features:
            batch = dict(batch)  # 浅拷贝，避免修改调用方
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return self.model(batch)[0]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """训练/验证前向：计算重建损失（L1）与可选的 KL 损失。"""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae and log_sigma_x2_hat is not None:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict
