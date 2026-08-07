"""lerobot_env_mujoco_lerobot — LeRobot third-party environment plugin.

LeRobot's ``register_third_party_plugins()`` discovers packages whose name
starts with ``lerobot_env_``, imports them, and triggers the
``@EnvConfig.register_subclass(...)`` registration in this package.

Usage:
    lerobot-eval --env.type=mujoco_lerobot --env.task=pick_place ...
"""

from __future__ import annotations

try:
    import lerobot  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this environment plugin."
    ) from exc

# Importing these modules registers the EnvConfig subclass.
from .lerobot_env_cfg import MujocoLerobotEnvConfig, build_features  # noqa: F401
from .lerobot_env import MujocoLerobotEnv  # noqa: F401

# 安装训练深度单位自动跟随补丁：训练时未显式指定 dataset.depth_output_unit 时，
# 自动跟随数据集 info.json 记录的单位（本项目为 "m"），使训练解码/stats 与评估
# env（米）一致，训练侧无需再显式指定单位。
from . import patches as _patches  # noqa: E402,F401

_patches.install_lerobot_depth_unit_patch()

__all__ = ["MujocoLerobotEnv", "MujocoLerobotEnvConfig", "build_features"]
