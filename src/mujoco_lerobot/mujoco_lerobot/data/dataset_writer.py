"""LeRobot 数据集写入器 — 流式写入（不缓存整集）。

使用 LeRobotDataset(streaming_encoding=True) + 后台编码线程：
  - 每帧通过流式回调直接写入，内存恒定
  - 编码由 lerobot 后台线程/进程池处理，不阻塞仿真主线程
  - 失败 episode 通过 discard 丢弃
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from lerobot.configs import DepthEncoderConfig, RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ..configs.dataset_config import DatasetConfig

os.environ.setdefault("FFMPEG_LOGLEVEL", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _name in ("lerobot", "datasets", "PIL", "torchvision", "ffmpeg", "av", "x265"):
    logging.getLogger(_name).setLevel(logging.WARNING)

log = logging.getLogger(__name__)


def build_video_encoders(
    config: "LeRobotDatasetConfig", dataset_cfg: DatasetConfig | None,
) -> tuple[DepthEncoderConfig, RGBEncoderConfig]:
    """按配置构建 rgb/depth 视频编码器（含画质 CRF 控制）。

    - depth：米制量化范围来自 ``dataset_cfg.depth_range``；
      ``depth_crf`` 未指定（None）→ lerobot 默认（无损 HEVC，lossless=1，不引入视频误差）；
      指定 → 去掉 lossless、改有损 HEVC，按该 CRF 压缩（文件显著变小但引入精度误差）。
    - rgb：``rgb_crf`` 未指定（None）→ lerobot 默认（CRF=30）；指定 → 用该 CRF。
    """
    # 深度量化范围（米制）
    depth_min, depth_max = (0.1, 2.0)
    if dataset_cfg is not None and dataset_cfg.depth_range:
        depth_min, depth_max = dataset_cfg.depth_range

    depth_crf = dataset_cfg.depth_crf if dataset_cfg is not None else None
    if depth_crf is None:
        depth_encoder = DepthEncoderConfig(depth_min=depth_min, depth_max=depth_max)
    else:
        depth_encoder = DepthEncoderConfig(
            depth_min=depth_min, depth_max=depth_max,
            crf=depth_crf, extra_options={},
        )

    rgb_crf = dataset_cfg.rgb_crf if dataset_cfg is not None else None
    rgb_kwargs: dict = dict(vcodec=config.vcodec, preset=config.preset, g=config.g)
    if rgb_crf is not None:
        rgb_kwargs["crf"] = rgb_crf
    rgb_encoder = RGBEncoderConfig(**rgb_kwargs)
    return depth_encoder, rgb_encoder


@contextlib.contextmanager
def _quiet_stderr():
    """抑制 ffmpeg/lerobot 的 stderr 噪音。"""
    import sys
    devnull = open(os.devnull, "w")
    old = os.dup(2)
    try:
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(old, 2)
        devnull.close()


@dataclass(slots=True)
class LeRobotDatasetConfig:
    repo_id: str
    root: str | Path
    fps: int
    cameras: list = field(default_factory=list)
    use_rgb: bool = True
    use_depth: bool = True
    streaming_encoding: bool = True
    vcodec: str = "auto"
    preset: str | None = None
    g: int | None = None            # GOP 大小；None=auto（NVENC 需要 g 足够大）
    batch_encoding_size: int = 1
    encoder_threads: int | None = 4
    encoder_queue_maxsize: int = 90
    image_writer_threads: int = 0
    image_writer_processes: int = 0

    def resolved_root(self) -> Path:
        return Path(self.root).expanduser().resolve()


class LeRobotDatasetWriter:
    def __init__(
        self,
        dataset: LeRobotDataset,
        config: LeRobotDatasetConfig,
        dataset_cfg: DatasetConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.cameras = config.cameras
        self._dataset_cfg = dataset_cfg

    @classmethod
    def create_new(
        cls,
        config: LeRobotDatasetConfig,
        dataset_cfg: DatasetConfig | None = None,
        *,
        overwrite: bool = False,
    ) -> "LeRobotDatasetWriter":
        root = config.resolved_root()
        if overwrite and root.exists():
            shutil.rmtree(root)

        depth_encoder, rgb_encoder = build_video_encoders(config, dataset_cfg)

        if dataset_cfg is not None:
            features = dataset_cfg.build_features(config.cameras)
        else:
            features = cls._default_features(config)

        kwargs = dict(
            repo_id=config.repo_id,
            root=root,
            fps=int(config.fps),
            features=features,
            use_videos=True,
            streaming_encoding=bool(config.streaming_encoding),
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
            batch_encoding_size=int(config.batch_encoding_size),
            encoder_threads=config.encoder_threads,
            encoder_queue_maxsize=int(config.encoder_queue_maxsize),
            image_writer_threads=int(config.image_writer_threads),
            image_writer_processes=int(config.image_writer_processes),
        )

        with _quiet_stderr():
            dataset = LeRobotDataset.create(**kwargs)
        return cls(dataset, config, dataset_cfg=dataset_cfg)

    # ── 流式回调 ───────────────────────────────────────

    @staticmethod
    def _default_features(config: "LeRobotDatasetConfig") -> dict[str, dict[str, Any]]:
        """无 dataset_cfg 时的兜底 features（仅相机，无状态）。"""
        features: dict[str, dict[str, Any]] = {}
        for cam in config.cameras:
            prefix = f"observation.images.{cam.name}"
            if config.use_rgb:
                features[f"{prefix}.rgb"] = {
                    "dtype": "video", "shape": (3, cam.height, cam.width),
                    "names": ["channel", "height", "width"],
                }
            if config.use_depth:
                features[f"{prefix}.depth"] = {
                    "dtype": "video", "shape": (1, cam.height, cam.width),
                    "names": ["channel", "height", "width"],
                    "info": {"is_depth_map": True},
                }
        return features

    def make_stream_callback(
        self, task_label: str = ""
    ) -> tuple[
        Callable[[dict[str, Any], np.ndarray], None],
        Callable[[], None],
        Callable[[], None],
    ]:
        """创建流式写入回调 + flush/discard。

        返回 (callback, flush, discard)：
          - callback(obs, action): 格式化并写入一帧
          - flush(): 保存当前 episode
          - discard(): 丢弃当前 episode 帧
        """
        ds = self.dataset

        def _callback(obs: dict[str, Any], action: np.ndarray) -> None:
            frame = self._format_frame(obs, action, task_label=task_label)
            ds.add_frame(frame)

        def _flush() -> None:
            if ds.has_pending_frames():
                ds.save_episode()

        def _discard() -> None:
            if ds.has_pending_frames():
                ds.clear_episode_buffer()

        return _callback, _flush, _discard

    def _format_frame(
        self,
        obs: dict[str, Any],
        action: np.ndarray,
        *,
        task_label: str,
    ) -> dict[str, Any]:
        """把采集器 flush 出的 obs/action 组帧为 LeRobotDataset 帧。"""
        frame: dict[str, Any] = {}

        state = obs.get("state", {})
        act = obs.get("action", {})

        if self._dataset_cfg is not None and self._dataset_cfg.flat_mode:
            # ── 扁平模式（ACT 兼容）：各源沿最后一维拼接 → (D,) ──
            if state:
                frame["observation.state"] = np.concatenate(
                    [np.asarray(v, dtype=np.float32).ravel() for v in state.values()]
                )
            if act:
                frame["action"] = np.concatenate(
                    [np.asarray(v, dtype=np.float32).ravel() for v in act.values()]
                )
            elif action is not None:
                frame["action"] = np.asarray(action, dtype=np.float32)
        else:
            # ── 多速率模式：每源独立 feature ──
            for key, val in state.items():
                frame[f"observation.{key}"] = np.asarray(val, dtype=np.float32)

            # action：各 action 源沿最后一维拼接 → (R, D)
            if act:
                parts = [np.asarray(v, dtype=np.float32) for v in act.values()]
                # 各源子采样数可能不同：统一到最多的子采样数，不足的重复最后一帧
                max_r = max(p.shape[0] for p in parts)
                aligned = []
                for p in parts:
                    if p.shape[0] < max_r:
                        pad = np.repeat(p[-1:], max_r - p.shape[0], axis=0)
                        p = np.concatenate([p, pad], axis=0)
                    aligned.append(p)
                frame["action"] = np.concatenate(aligned, axis=-1)
            elif action is not None:
                frame["action"] = np.asarray(action, dtype=np.float32)

        images = obs.get("images", {})
        for cam in self.cameras:
            img = images.get(cam.name)
            if img is None:
                continue
            if self.config.use_rgb and img.get("rgb") is not None:
                frame[f"observation.images.{cam.name}.rgb"] = np.ascontiguousarray(
                    img["rgb"], dtype=np.uint8
                )
            if self.config.use_depth and img.get("depth") is not None:
                d = np.asarray(img["depth"], dtype=np.float32)
                frame[f"observation.images.{cam.name}.depth"] = d[np.newaxis, ...]

        frame["task"] = task_label
        return frame

    # ── 收尾 ───────────────────────────────────────────

    def finalize(self) -> None:
        self.dataset.finalize()

    def close(self) -> None:
        self.finalize()
