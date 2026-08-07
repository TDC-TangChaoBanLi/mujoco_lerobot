"""Teacher 配置加载 — 从 configs/teachers/*.yaml 读取 scripted teacher 参数。

teacher 状态机中的硬编码常量全部迁移到 yaml，加载为 dataclass 后注入 teacher。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config_loader import resolve_config_path


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _f(d: dict[str, Any], key: str, default: float) -> float:
    return float(d.get(key, default))


def _i(d: dict[str, Any], key: str, default: int) -> int:
    return int(d.get(key, default))


def _tup(d: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    v = d.get(key)
    if v is None:
        return default
    return (float(v[0]), float(v[1]))


@dataclass
class TeacherThresh:
    approach_pos: float = 0.03
    approach_rot: float = 0.1
    grasp_dist: float = 0.08
    drop_dist: float = 0.10
    place_dist: float = 0.10
    gripper_wait: int = 50
    settle_wait: int = 30
    max_retries: int = 3

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TeacherThresh":
        return cls(
            approach_pos=_f(d, "approach_pos", cls.approach_pos),
            approach_rot=_f(d, "approach_rot", cls.approach_rot),
            grasp_dist=_f(d, "grasp_dist", cls.grasp_dist),
            drop_dist=_f(d, "drop_dist", cls.drop_dist),
            place_dist=_f(d, "place_dist", cls.place_dist),
            gripper_wait=_i(d, "gripper_wait", cls.gripper_wait),
            settle_wait=_i(d, "settle_wait", cls.settle_wait),
            max_retries=_i(d, "max_retries", cls.max_retries),
        )


@dataclass
class TeacherHeights:
    above: float = 0.12
    lift: float = 0.10
    place: float = 0.08
    retreat: float = 0.15

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TeacherHeights":
        return cls(
            above=_f(d, "above", cls.above),
            lift=_f(d, "lift", cls.lift),
            place=_f(d, "place", cls.place),
            retreat=_f(d, "retreat", cls.retreat),
        )


@dataclass
class GripperCmd:
    open: float = 0.0
    close: float = 0.8

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GripperCmd":
        return cls(
            open=_f(d, "open", cls.open),
            close=_f(d, "close", cls.close),
        )


@dataclass
class PickPlaceTeacherConfig:
    type: str = "PickPlaceTeacher"
    cube: str = "cube"
    plate: str = "plate"
    grasp_quat: tuple[float, float, float, float] = (0.0, 0.7071, 0.7071, 0.0)
    thresh: TeacherThresh = field(default_factory=TeacherThresh)
    heights: TeacherHeights = field(default_factory=TeacherHeights)
    gripper: GripperCmd = field(default_factory=GripperCmd)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PickPlaceTeacherConfig":
        raw = _load_yaml(resolve_config_path(path))
        objects = raw.get("objects", {}) or {}
        quat = raw.get("grasp_quat", [0.0, 0.7071, 0.7071, 0.0])
        return cls(
            type=str(raw.get("type", cls.type)),
            cube=str(objects.get("cube", "cube")),
            plate=str(objects.get("plate", "plate")),
            grasp_quat=tuple(float(v) for v in quat),
            thresh=TeacherThresh.from_dict(raw.get("thresh", {}) or {}),
            heights=TeacherHeights.from_dict(raw.get("heights", {}) or {}),
            gripper=GripperCmd.from_dict(raw.get("gripper", {}) or {}),
        )


@dataclass
class DualPickPlaceTeacherConfig:
    type: str = "DualPickPlaceTeacher"
    heights: TeacherHeights = field(default_factory=TeacherHeights)
    table_z: float = 0.65
    center_x_range: tuple[float, float] = (-0.05, 0.05)
    center_y_half_range: tuple[float, float] = (-0.2, -0.3)
    retreat_center_a: tuple[float, float] = (-0.3, 0.3)
    retreat_center_b: tuple[float, float] = (0.3, -0.3)
    retreat_radius: float = 0.1
    grasp_euler_a: tuple[float, float, float] = (3.14159, 0.0, 1.5708)
    grasp_euler_b: tuple[float, float, float] = (3.14159, 0.0, -1.5708)
    thresh: TeacherThresh = field(default_factory=TeacherThresh)
    gripper: GripperCmd = field(default_factory=GripperCmd)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DualPickPlaceTeacherConfig":
        raw = _load_yaml(resolve_config_path(path))
        heights = TeacherHeights.from_dict(raw.get("heights", {}) or {})
        table_z = _f(raw.get("heights", {}) or {}, "table_z", cls.table_z)
        center = raw.get("center", {}) or {}
        retreat = raw.get("retreat", {}) or {}
        euler = raw.get("grasp_euler", {}) or {}
        ea = euler.get("a", [3.14159, 0.0, 1.5708])
        eb = euler.get("b", [3.14159, 0.0, -1.5708])
        return cls(
            type=str(raw.get("type", cls.type)),
            heights=heights,
            table_z=table_z,
            center_x_range=_tup(center, "x_range", cls.center_x_range),
            center_y_half_range=_tup(center, "y_half_range", cls.center_y_half_range),
            retreat_center_a=_tup(retreat, "center_a", cls.retreat_center_a),
            retreat_center_b=_tup(retreat, "center_b", cls.retreat_center_b),
            retreat_radius=_f(retreat, "radius", cls.retreat_radius),
            grasp_euler_a=tuple(float(v) for v in ea),
            grasp_euler_b=tuple(float(v) for v in eb),
            thresh=TeacherThresh.from_dict(raw.get("thresh", {}) or {}),
            gripper=GripperCmd.from_dict(raw.get("gripper", {}) or {}),
        )


# ── 统一加载入口 ──────────────────────────────────────


def load_teacher_config(path: str | Path) -> Any:
    """根据 yaml 中的 type 加载对应 teacher 配置。

    通过 teacher 注册表自动发现配置类，支持外部任务在各自模块注册后
    直接在此加载，无需修改本文件。
    """
    from ..data.teachers import discover_teachers, get_teacher_config_class

    discover_teachers()  # 确保所有内置 teacher 已注册
    raw = _load_yaml(resolve_config_path(path))
    ttype = str(raw.get("type", ""))
    if not ttype:
        raise ValueError(f"teacher 配置缺少 type 字段（{path}）")
    cfg_cls = get_teacher_config_class(ttype)
    return cfg_cls.from_yaml(path)
