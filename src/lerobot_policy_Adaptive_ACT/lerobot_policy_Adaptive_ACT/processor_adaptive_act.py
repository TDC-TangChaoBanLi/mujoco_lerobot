"""processor_adaptive_act — 自适应 ACT 策略的 pre/post 处理器。

直接复用 LeRobot 默认处理器：对全部输入（VISUAL / STATE / ENV）做 MEAN_STD
归一化（依赖 dataset_stats），对输出（ACTION）做反归一化并移回 CPU。
因此模型训练/推理的所有输入输出均被归一化，满足任务无关与归一化要求。
"""

from __future__ import annotations

from typing import Any

import torch

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    make_default_pre_post_processors,
)

from .configuration_adaptive_act import AdaptiveACTConfig


def make_adaptive_act_pre_post_processors(
    config: AdaptiveACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """创建自适应 ACT 的 pre/post 处理器（默认 MEAN_STD 归一化）。"""
    return make_default_pre_post_processors(
        config, dataset_stats, normalizer_device=config.device
    )
