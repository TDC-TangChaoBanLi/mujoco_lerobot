"""ExampleMLPConfig — 消费 mujoco-lerobot 数据集格式观测的策略配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedConfig as _PreTrainedConfigRegistry  # noqa: F401
from lerobot.utils.constants import ACTION


def _make_input_features(
    state_key: str = "observation.state.joint.position",
    state_shape: tuple[int, int] = (3, 7),
    camera_names: tuple[str, ...] = ("realsense_link_CAMERA",),
) -> dict[str, PolicyFeature]:
    features: dict[str, PolicyFeature] = {
        state_key: PolicyFeature(type=FeatureType.STATE, shape=state_shape),
    }
    for cam in camera_names:
        features[f"observation.images.{cam}.rgb"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 480, 640)
        )
        features[f"observation.images.{cam}.depth"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(1, 480, 640)
        )
    return features


@_PreTrainedConfigRegistry.register_subclass("mujoco_lerobot_example")
@dataclass
class ExampleMLPConfig(PreTrainedConfig):
    """示例 MLP 策略配置。

    默认输入与 dataset_pick_place.yaml 一致；评估时可用 --policy.path 指向训练好的
    checkpoint，其 config.json 中保存了真实的 input_features / output_features。
    """

    n_obs_steps: int = 1
    n_action_steps: int = 1
    hidden_dim: int = 128

    def __post_init__(self) -> None:
        super().__post_init__()
        # 训练/评估时由数据集或 eval 环境动态确定；此默认值仅用于从零初始化演示
        if not self.input_features:
            self.input_features = _make_input_features()
        if not self.output_features:
            self.output_features = {
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))
            }

    # ── PreTrainedConfig 抽象方法（最小实现）──────────────

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.n_action_steps))

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> None:
        return None

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not any(
            ft.type is FeatureType.STATE for ft in self.input_features.values()
        ):
            raise ValueError("ExampleMLPPolicy 需要至少一个 STATE 输入特征")
