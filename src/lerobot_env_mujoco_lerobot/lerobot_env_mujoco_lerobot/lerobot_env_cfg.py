"""LeRobot EnvConfig 注册 — mujoco_lerobot 评估环境。

用法:
    lerobot-eval --env.type=mujoco_lerobot \\
        --env.task=pick_place \\
        --env.dataset_config=configs/dataset/dataset_pick_place.yaml \\
        --policy.path=...
    # 可视化评估（打开 MuJoCo viewer）:
        --env.use_viewer=true

环境产出 gym 风格键（state.* / images.*），lerobot 的通用 preprocess_observation
会自动补上 `observation.` 前缀，features_map 亦将键映射到 observation.*。
任何在该数据集格式上训练的策略可直接评估；其他策略可覆盖 get_env_processors
提供外部观测数据处理器。

本环境插件是任务无关的：task / dataset_config 均由 CLI 或配置传入。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.pipeline import ObservationProcessorStep
from lerobot.lerobot_types import EnvTransition
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from mujoco_lerobot.configs.config_loader import load_scene_config
from mujoco_lerobot.configs.dataset_config import DatasetConfig

from .lerobot_env import MujocoLerobotEnv


_MM_PER_METRE = 1000.0


class ScaleRgbImagesProcessorStep(ObservationProcessorStep):
    """将 rgb 图像从 uint8 [0,255] 缩放到 float [0,1]。

    训练时 LeRobot 在 dataloader 后把视频帧 `uint8 / 255 → [0,1]` 再交给
    policy preprocessor 做 MEAN_STD 归一化；而评估时 env 直接产出 uint8 图像，
    通用 preprocess_observation 对 `images.*` 键只转 float 不缩放到 [0,1]。
    此步骤补齐 `observation.images.<cam>.rgb` 的缩放，使推理输入分布与训练一致。
    """

    def observation(self, observation):
        processed = dict(observation)
        for key, value in observation.items():
            if key.endswith(".rgb"):
                processed[key] = value / 255.0
        return processed

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # ObservationProcessorStep.__call__ 会拷贝 transition 并调用 observation()
        return super().__call__(transition)

    def transform_features(self, features):
        # 缩放不改变 shape/dtype 描述，原样返回
        return features


class ScaleDepthToOutputUnitProcessorStep(ObservationProcessorStep):
    """把 env 产出的米制 depth 转到训练使用的 ``depth_output_unit``。

    训练时 LeRobot 以 ``dataset.depth_output_unit`` 解码深度视频（本项目统一设为
    ``"m"``，与 info.json 记录的 ``depth_unit`` 一致），stats 亦为米；评估 env 直接
    产出物理米值。此步骤按目标单位转换：``"m"`` 保持不变，``"mm"`` 乘以 1000，
    使评估输入表示与训练一致——否则策略 Normalizer（stats 为对应单位）收到的深度
    分布错误，深度通道在评估时退化（"失明"）。
    """

    def __init__(self, depth_output_unit: str = "m"):
        super().__init__()
        if depth_output_unit not in ("m", "mm"):
            raise ValueError(
                f"depth_output_unit must be 'm' or 'mm', got {depth_output_unit!r}"
            )
        self.depth_output_unit = depth_output_unit
        self._factor = _MM_PER_METRE if depth_output_unit == "mm" else 1.0

    def observation(self, observation):
        if self._factor == 1.0:
            return observation
        processed = dict(observation)
        for key, value in observation.items():
            if key.endswith(".depth"):
                processed[key] = value * self._factor
        return processed

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return super().__call__(transition)

    def transform_features(self, features):
        # 缩放不改变 shape/dtype 描述，原样返回
        return features


_ENV_TYPE = "mujoco_lerobot"


def build_features(
    task: str, dataset_config: str,
) -> tuple[dict[str, PolicyFeature], dict[str, str]]:
    """从 dataset + 场景配置构建 features / features_map。

    features 的键与环境产出的 gym 键一致（state.* / images.*）；
    features_map 将其映射为 LeRobot 标准 observation.* 键。
    """
    scene_cfg = load_scene_config(task)
    dataset_cfg = DatasetConfig.from_yaml(dataset_config)

    features: dict[str, PolicyFeature] = {
        ACTION: PolicyFeature(
            type=FeatureType.ACTION, shape=(scene_cfg.action_dim,)
        ),
    }
    features_map: dict[str, str] = {ACTION: ACTION}

    if dataset_cfg.flat_mode:
        # 扁平模式：单个 observation.state (state_dim,)（ACT 兼容）
        features["state"] = PolicyFeature(
            type=FeatureType.STATE, shape=(dataset_cfg.state_dim,)
        )
        features_map["state"] = OBS_STATE
    else:
        for src in dataset_cfg.state_sources():
            features[src.name] = PolicyFeature(
                type=FeatureType.STATE, shape=(src.num_subs, src.dim_per_sub)
            )
            features_map[src.name] = f"{OBS_STATE}.{src.name}"

    for cam in scene_cfg.cameras:
        for suffix, shape in (
            ("rgb", (3, cam.height, cam.width)),
            ("depth", (1, cam.height, cam.width)),
        ):
            key = f"images.{cam.name}.{suffix}"
            features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
            features_map[key] = f"{OBS_IMAGES}.{cam.name}.{suffix}"

    return features, features_map


@EnvConfig.register_subclass(_ENV_TYPE)
@dataclass
class MujocoLerobotEnvConfig(EnvConfig):
    """MuJoCo 评估环境配置（dataset 配置驱动，任务无关）。"""

    task: str = "pick_place"
    dataset_config: str = "configs/dataset/dataset_pick_place.yaml"
    # 是否打开 MuJoCo viewer（可视化评估）；默认 False（无头，render() 返回 RGB 供录视频）
    use_viewer: bool = False
    max_episode_steps: int | None = None
    fps: int = 30
    # 深度输出单位：env 产出米制 depth，此值须与训练 `dataset.depth_output_unit`
    # 一致（统一设为 "m"，与 info.json 记录的 depth_unit 一致）。"mm" 表示在
    # env preprocessor 中把米×1000 转毫米以匹配 mm 训练。
    depth_output_unit: str = "m"

    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 由 dataset 配置动态生成 features / features_map
        feat, feat_map = build_features(self.task, self.dataset_config)
        self.features = feat
        self.features_map = feat_map

    @property
    def gym_kwargs(self) -> dict:
        return {
            "task_name": self.task,
            "dataset_config": self.dataset_config,
            "use_viewer": self.use_viewer,
            "max_episode_steps": self.max_episode_steps,
        }

    def create_envs(
        self,
        n_envs: int,
        use_async_envs: bool = False,
    ) -> dict[str, dict[int, gym.vector.VectorEnv]]:
        """创建 VectorEnv。use_viewer=True 时强制单环境（viewer 只能一个）。"""
        use_human = self.use_viewer
        use_async = use_async_envs and n_envs > 1 and not use_human
        env_cls = (
            gym.vector.AsyncVectorEnv if use_async else gym.vector.SyncVectorEnv
        )

        kwargs = self.gym_kwargs

        def _make_one():
            return MujocoLerobotEnv(**kwargs)

        vec = env_cls([_make_one for _ in range(n_envs)])
        return {self.type: {0: vec}}

    def get_env_processors(self):
        """评估侧观测处理器。

        preprocessor 包含两个缩放步骤，使 env 产出的观测与训练分布一致：
        1. rgb：uint8 [0,255] → [0,1]（对齐训练 dataloader 的 /255）；
        2. depth：按 ``self.depth_output_unit`` 转换（默认 "m" 不变，env 已为米；
           "mm" 时 ×1000），与训练 ``dataset.depth_output_unit`` 对齐。
        否则策略 preprocessor（MEAN_STD，stats 基于对应表示）收到的输入范围错误，
        导致策略退化。其他策略如需适配，可在此追加自定义处理步骤。
        """
        preprocessor = PolicyProcessorPipeline(
            steps=[
                ScaleRgbImagesProcessorStep(),
                ScaleDepthToOutputUnitProcessorStep(self.depth_output_unit),
            ]
        )
        postprocessor = PolicyProcessorPipeline(steps=[])
        return preprocessor, postprocessor
