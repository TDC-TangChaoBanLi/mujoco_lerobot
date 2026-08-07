"""lerobot_policy_mujoco_lerobot_example — 示例策略插件。

一个消费 mujoco-lerobot 数据集格式观测（observation.state.* / observation.images.*）
的最小 MLP 策略，用于演示 lerobot-eval 端到端评估流程。
"""

from __future__ import annotations

try:
    import lerobot  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy plugin."
    ) from exc

from .configuration_example import ExampleMLPConfig
from .modeling_example import ExampleMLPPolicy

__all__ = ["ExampleMLPConfig", "ExampleMLPPolicy"]
