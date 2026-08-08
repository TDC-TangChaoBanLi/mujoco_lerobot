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

__all__ = ["MujocoLerobotEnv", "MujocoLerobotEnvConfig", "build_features"]
