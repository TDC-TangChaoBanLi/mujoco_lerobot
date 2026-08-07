"""lerobot_policy_Adaptive_ACT — 自适应 ACT 策略插件。

基于原生 ACT（Action Chunking Transformer），新增：
1. 任意图像通道数输入（1/2/3/4/... 通道），非 3 通道时仅替换 resnet 的 conv1，
   其余层保留 ImageNet 预训练权重；
2. 相机可按分组共享或独立使用不同的 resnet18 backbone
   （``camera_backbone_groups``，未指定的相机自动归入共享默认组）；
3. 输入图像通道数可从数据集自动识别（``image_channels=None``）；
4. 全部输入/输出经 LeRobot 默认 MEAN_STD 归一化；
5. 模型结构基于纯 ``torch.nn.Transformer`` 构建，任务无关。

注册为 LeRobot 策略类型 ``adaptive_act``：
    --policy.type=adaptive_act
"""

from __future__ import annotations

try:
    import lerobot  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy plugin."
    ) from exc

from .configuration_adaptive_act import AdaptiveACTConfig
from .modeling_adaptive_act import AdaptiveACTPolicy

__all__ = ["AdaptiveACTConfig", "AdaptiveACTPolicy"]
