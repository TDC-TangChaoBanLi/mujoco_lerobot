"""数据集配置加载 — 解析 dataset_*.yaml 的多速率记录格式。

支持嵌套 recode_scale，各数据源以不同频率采样，打包进同一 LeRobot 帧。

use_recode_scale 全局开关：
  true  = 启用 recode_scale 扩展：非相机数据以 recode_hz × recode_scale 采样，
          一帧内 shape (R, D)；
  false = 关闭：所有非相机数据按 recode_hz 记录（shape 不扩展），忽略所有
          recode_scale 字段。

相机数据固定按 recode_hz 记录（不扩展）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT
from .config_loader import resolve_config_path

_COMP_SUFFIXES = {3: ["_x", "_y", "_z"], 4: ["_w", "_x", "_y", "_z"]}
_SENSOR_DIM = {"force": 3, "torque": 3, "gyro": 3, "accelerometer": 3}
_SITE_DIM = {"position": 3, "euler": 3, "quat": 4}


@dataclass
class DataSource:
    """单个数据源的规格。"""

    name: str                              # "state.joint.position"
    dim_per_sub: int                       # 每子采样维度 (14 关节 → 14)
    num_subs: int                          # 子采样数 (recode_scale=3 → 3)
    source_type: str                       # "joint_pos" | "joint_vel" | "joint_effort" |
                                           # "sensor.force" | "site_position" | ...
    source_names: list[str]                # 展开后的显示名（对齐最后一维）
    read_names: list[str] = field(default_factory=list)  # MuJoCo 原始名（传感器用）
    frame_site: str = "world"             # site 位姿的参考系（world / 其他 site 名）
    pose_type: str = "absolute"           # site 位姿类型：absolute / relative / velocity

    @property
    def total_dim(self) -> int:
        """该源在帧内的总维度（num_subs × dim_per_sub）。"""
        return self.num_subs * self.dim_per_sub


@dataclass
class DatasetConfig:
    """数据集记录配置。"""

    recode_hz: float = 30.0
    max_scale: int = 1
    use_recode_scale: bool = True
    sources: list[DataSource] = field(default_factory=list)
    camera_config_file: str = ""
    depth_range: tuple[float, float] = (0.1, 2.0)

    # ── 加载 ──────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetConfig":
        full = resolve_config_path(path)
        if not full.exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {full}")
        with open(full, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        recode_hz = float(raw.get("recode_hz", 30.0))
        use_recode_scale = bool(raw.get("use_recode_scale", True))

        sources: list[DataSource] = []
        cls._walk_tree(raw.get("state", {}), "state", sources, 1, use_recode_scale)
        cls._walk_tree(raw.get("action", {}), "action", sources, 1, use_recode_scale)

        max_scale = max((s.num_subs for s in sources), default=1)

        # 相机配置：记录 scene_config_file + depth_range
        camera_file = ""
        depth_range = (0.1, 2.0)
        state_cam = raw.get("state", {}).get("camera", {})
        if isinstance(state_cam, dict):
            camera_file = str(state_cam.get("scene_config_file", ""))
            dr = state_cam.get("depth_range")
            if dr and len(dr) == 2:
                depth_range = (float(dr[0]), float(dr[1]))

        return cls(
            recode_hz=recode_hz,
            max_scale=max_scale,
            use_recode_scale=use_recode_scale,
            sources=sources,
            camera_config_file=camera_file,
            depth_range=depth_range,
        )

    @classmethod
    def _walk_tree(
        cls,
        node: Any,
        prefix: str,
        sources: list[DataSource],
        parent_scale: int,
        use_recode_scale: bool,
    ) -> None:
        """递归遍历 YAML 树，收集叶子节点为 DataSource。"""
        if not isinstance(node, dict):
            return

        if use_recode_scale:
            scale = parent_scale * int(node.get("recode_scale", 1))
        else:
            scale = 1

        # ── 叶子 1：joint_names（position / velocity / effort）──
        if "joint_names" in node:
            names = [str(n) for n in node["joint_names"]]
            sub = prefix.rsplit(".", 1)[-1]
            stype = {
                "position": "joint_pos",
                "velocity": "joint_vel",
                "effort": "joint_effort",
            }.get(sub, "joint_pos")
            cls._add_source(
                sources, prefix, len(names), scale,
                stype, names, read_names=names,
            )
            return

        # ── 叶子 2：sensor_names ──
        if "sensor_names" in node:
            sensor_type = prefix.rsplit(".", 1)[-1]
            dim_per = _SENSOR_DIM.get(sensor_type, 1)
            names = [str(n) for n in node["sensor_names"]]
            expanded = cls._expand_names(names, dim_per)
            cls._add_source(
                sources, prefix, len(names) * dim_per, scale,
                f"sensor.{sensor_type}", expanded, read_names=names,
            )
            return

        # ── 叶子 3：site_names（position / euler / quat，可带 frame_site / type）──
        if "site_names" in node:
            suffix = prefix.rsplit(".", 1)[-1]  # position / euler / quat
            dim_per = _SITE_DIM.get(suffix, 3)
            names = [str(n) for n in node["site_names"]]
            expanded = cls._expand_names(names, dim_per)
            cls._add_source(
                sources, prefix, len(names) * dim_per, scale,
                f"site_{suffix}", expanded, read_names=names,
                frame_site=str(node.get("frame_site", "world")),
                pose_type=str(node.get("type", "absolute")),
            )
            return

        # ── 相机叶子：无数据维度，仅标记存在 ──
        if "scene_config_file" in node:
            return

        # ── 递归子节点 ──
        for key, val in node.items():
            if key in (
                "recode_scale", "joint_names", "sensor_names",
                "site_names", "scene_config_file", "frame_site", "type",
            ):
                continue
            sub_prefix = f"{prefix}.{key}" if prefix else key
            cls._walk_tree(val, sub_prefix, sources, scale, use_recode_scale)

    @staticmethod
    def _expand_names(names: list[str], dim_per: int) -> list[str]:
        suffixes = _COMP_SUFFIXES.get(
            dim_per, [f"_{i}" for i in range(dim_per)]
        )
        return [n + s for n in names for s in suffixes]

    @staticmethod
    def _add_source(
        sources: list[DataSource],
        name: str,
        dim: int,
        scale: int,
        source_type: str,
        names: list[str],
        read_names: list[str] | None = None,
        frame_site: str = "world",
        pose_type: str = "absolute",
    ) -> None:
        sources.append(
            DataSource(
                name=name,
                dim_per_sub=dim,
                num_subs=max(1, scale),
                source_type=source_type,
                source_names=names,
                read_names=read_names if read_names is not None else names,
                frame_site=frame_site,
                pose_type=pose_type,
            )
        )

    # ── 派生属性 ──────────────────────────────────────

    @property
    def flat_mode(self) -> bool:
        """use_recode_scale=false 时，产出 LeRobot 标准扁平格式（ACT 兼容）。"""
        return not self.use_recode_scale

    @property
    def frame_interval_s(self) -> float:
        return 1.0 / self.recode_hz

    @property
    def sample_interval_s(self) -> float:
        """子采样间隔（秒）：1 / (recode_hz × max_scale)。"""
        return 1.0 / (self.recode_hz * self.max_scale)

    def state_sources(self) -> list[DataSource]:
        return [s for s in self.sources if s.name.startswith("state.")]

    def action_sources(self) -> list[DataSource]:
        return [s for s in self.sources if s.name.startswith("action.")]

    @property
    def state_dim(self) -> int:
        """所有 state 源的总维度（num_subs × dim_per_sub 之和）。"""
        return sum(s.total_dim for s in self.state_sources())

    @property
    def action_dim(self) -> int:
        """所有 action 源的总维度。"""
        return sum(s.total_dim for s in self.action_sources())

    @property
    def action_scale(self) -> int:
        """action 的最大子采样数（决定 action feature 的 R）。"""
        acts = self.action_sources()
        return max((s.num_subs for s in acts), default=1)

    def action_total_dim_per_sub(self) -> int:
        """所有 action 源单帧（一个子采样）的总维度。"""
        return sum(s.dim_per_sub for s in self.action_sources())

    # ── LeRobot features ──────────────────────────────

    def build_features(self, cameras: list[Any]) -> dict[str, dict[str, Any]]:
        """构建 LeRobotDataset features dict。

        flat 模式（use_recode_scale=false）：
            所有 state 源拼接为单个 observation.state (state_dim,)；
            action 为单个 (action_dim,)。可直接用于 LeRobot 内置 ACT。
        多速率模式：每个 state 源独立 feature，保留真实子采样数：
            observation.state.joint.position → (R, D)
        相机 → observation.images.<cam>.rgb / .depth
        action 统一为 "action": (max_subs, total_dim_per_sub)
        """
        if self.flat_mode:
            return self._build_flat_features(cameras)

        features: dict[str, dict[str, Any]] = {}

        for src in self.state_sources():
            if src.source_type.startswith("camera"):
                continue
            features[f"observation.{src.name}"] = {
                "dtype": "float32",
                "shape": (src.num_subs, src.dim_per_sub),
                "names": src.source_names if src.source_names else None,
            }

        # action
        act_names: list[str] = []
        for src in self.action_sources():
            act_names.extend(src.source_names)
        features["action"] = {
            "dtype": "float32",
            "shape": (self.action_scale, self.action_total_dim_per_sub()),
            "names": act_names if act_names else None,
        }

        # 相机
        for cam in cameras:
            prefix = f"observation.images.{cam.name}"
            features[f"{prefix}.rgb"] = {
                "dtype": "video",
                "shape": (3, cam.height, cam.width),
                "names": ["channel", "height", "width"],
            }
            features[f"{prefix}.depth"] = {
                "dtype": "video",
                "shape": (1, cam.height, cam.width),
                "names": ["channel", "height", "width"],
                "info": {"is_depth_map": True},
            }
        return features

    def _build_flat_features(self, cameras: list[Any]) -> dict[str, dict[str, Any]]:
        """LeRobot 标准扁平格式：observation.state (D,) + action (D,) + 相机。

        与 LeRobot 内置 ACT 等策略的默认 input_features 完全兼容。
        """
        features: dict[str, dict[str, Any]] = {}

        state_names: list[str] = []
        for src in self.state_sources():
            state_names.extend(src.source_names)
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (self.state_dim,),
            "names": state_names if state_names else None,
        }

        act_names: list[str] = []
        for src in self.action_sources():
            act_names.extend(src.source_names)
        features["action"] = {
            "dtype": "float32",
            "shape": (self.action_dim,),
            "names": act_names if act_names else None,
        }

        for cam in cameras:
            prefix = f"observation.images.{cam.name}"
            features[f"{prefix}.rgb"] = {
                "dtype": "video",
                "shape": (3, cam.height, cam.width),
                "names": ["channel", "height", "width"],
            }
            features[f"{prefix}.depth"] = {
                "dtype": "video",
                "shape": (1, cam.height, cam.width),
                "names": ["channel", "height", "width"],
                "info": {"is_depth_map": True},
            }
        return features

    def validate_names(self, model: Any) -> list[str]:
        """校验所有数据源名称在 MuJoCo 模型中存在，返回警告列表。"""
        warnings: list[str] = []
        mj_joints = {model.joint(i).name for i in range(model.njnt)}
        mj_sensors = {model.sensor(i).name for i in range(model.nsensor)}
        mj_sites = {model.site(i).name for i in range(model.nsite)}

        for src in self.sources:
            if src.source_type == "joint_pos":
                for n in src.read_names:
                    if n not in mj_joints:
                        warnings.append(
                            f"[dataset] 关节 {n!r}（源 {src.name}）不存在于 mjcf 模型"
                        )
            elif src.source_type.startswith("sensor."):
                for n in src.read_names:
                    if n not in mj_sensors:
                        warnings.append(
                            f"[dataset] 传感器 {n!r}（源 {src.name}）不存在于 mjcf 模型"
                        )
            elif src.source_type.startswith("site_"):
                for n in src.read_names:
                    if n not in mj_sites:
                        warnings.append(
                            f"[dataset] site {n!r}（源 {src.name}）不存在于 mjcf 模型"
                        )
        return warnings
