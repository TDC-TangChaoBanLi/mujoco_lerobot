"""processor_example — 示例策略的 pre/post 处理器。

环境直接产出 LeRobot 数据集格式观测，策略直接消费，因此 pre 处理器恒等；
post 处理器使用标准的 policy-action 转换器（raw tensor ↔ transition）保持恒等。
实际策略可在此实现归一化 / key 重命名。
"""

from __future__ import annotations

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)


def make_mujoco_lerobot_example_pre_post_processors(
    config, **kwargs,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """返回 (preprocessor, postprocessor)，均为恒等。"""
    preprocessor = PolicyProcessorPipeline(steps=[])

    # post 处理 raw action tensor：包成 transition → 恒等步 → 解包
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
