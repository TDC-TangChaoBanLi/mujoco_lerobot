"""lerobot_env_mujoco_lerobot — LeRobot third-party environment plugin.

LeRobot's ``register_third_party_plugins()`` discovers packages whose name
starts with ``lerobot_env_``, imports them, and triggers the
``@EnvConfig.register_subclass(...)`` registration in this package.

Usage:
    lerobot-eval --env.type=mujoco_lerobot --env.task=pick_place ...
"""

from __future__ import annotations

# 无头环境时在首个 `import mujoco` 之前切到 EGL 离屏后端（见 gl_setup.py）。
from mujoco_lerobot.gl_setup import configure as _configure_headless_gl

_configure_headless_gl()

try:
    import lerobot  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this environment plugin."
    ) from exc

# Importing these modules registers the EnvConfig subclass.
from .lerobot_env_cfg import MujocoLerobotEnvConfig, build_features  # noqa: F401
from .lerobot_env import MujocoLerobotEnv  # noqa: F401


def _patch_eval_recording_depth_features() -> None:
    """让 lerobot-eval 录制时正确识别深度 feature（``is_depth_map`` + ``depth_unit``）。

    lerobot 的 ``_env_features_to_dataset_features`` 只根据 PolicyFeature 的
    type/shape 生成数据集 features，深度 feature 缺少 ``info.is_depth_map``
    标记，导致 eval 录制时深度图被当作普通 RGB 图像存为 PNG——float32 深度
    无法写入 PNG（``write_image`` 静默吞掉异常），帧全部丢失，随后
    ``compute_episode_stats`` 加载不存在的帧报 FileNotFoundError。
    此 patch 为 ``.depth`` 后缀的 feature 补上 ``is_depth_map`` 与
    ``depth_unit`` 标记，使深度图以 TIFF 帧存储并正确编码为深度视频。
    """
    try:
        from lerobot.scripts import lerobot_eval
    except ImportError:
        return
    if getattr(lerobot_eval, "_env_features_to_dataset_features_patched", False):
        return

    _original = lerobot_eval._env_features_to_dataset_features

    def _patched(env_features):
        features = _original(env_features)
        for key, ft in features.items():
            if key.endswith(".depth"):
                ft["info"] = {"is_depth_map": True, "depth_unit": "m"}
        return features

    lerobot_eval._env_features_to_dataset_features = _patched
    lerobot_eval._env_features_to_dataset_features_patched = True


_patch_eval_recording_depth_features()

__all__ = ["MujocoLerobotEnv", "MujocoLerobotEnvConfig", "build_features"]
