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


# ── PushT 遥操作 Teacher 配置 ─────────────────────────
#
# PushT 鼠标遥操作的所有参数（窗口 / TCP / 可推动带 / 灵敏度 / 成功阈值）
# 扁平化在单一 PushTTeacherConfig 中，对应 yaml 的嵌套结构：
#   window: {width, height, workspace: {x_range, y_range}}
#   tcp:    {z_min, z_max, initial_z}
#   push:   {z_min, z_max}
#   sens:   {mouse, wheel, step}
#   success:{dist, yaw}


@dataclass
class PushTTeacherConfig:
    """PushT 鼠标遥操作 teacher 配置（唯一的数据类，扁平化嵌套 yaml 参数）。"""

    type: str = "PushTTeacher"
    t_obj: str = "t_obj"
    t_target: str = "t_target"
    ee_site: str = "_tcp"
    grasp_quat: tuple[float, float, float, float] = (0.0, 0.7071, 0.7071, 0.0)
    gripper_cmd: float = 0.8
    # window（2D 俯视窗口像素尺寸 + 展示的世界范围）
    window_width: int = 800
    window_height: int = 600
    workspace_x: tuple[float, float] = (-0.6, 0.6)
    workspace_y: tuple[float, float] = (-0.45, 0.45)
    # tcp（TCP 高度范围）
    tcp_z_min: float = 0.62
    tcp_z_max: float = 0.85
    tcp_initial_z: float = 0.68
    # push（可推动高度带：TCP z 落在此范围内可在水平面推动 T 物块）
    push_z_min: float = 0.66
    push_z_max: float = 0.72
    # sens（鼠标 / 滚轮灵敏度初值与调节步长；invert_x/y 为鼠标方向反转修正）
    sens_mouse: float = 1.5
    sens_wheel: float = 0.008
    sens_step: float = 0.1
    sens_invert_x: bool = False
    sens_invert_y: bool = False
    # success（成功判定阈值：T 中心到目标距离 + yaw 差）
    success_dist: float = 0.05
    success_yaw: float = 0.35

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PushTTeacherConfig":
        raw = _load_yaml(resolve_config_path(path))
        objects = raw.get("objects", {}) or {}
        quat = raw.get("grasp_quat", [0.0, 0.7071, 0.7071, 0.0])
        window = raw.get("window", {}) or {}
        ws = window.get("workspace", {}) or {}
        tcp = raw.get("tcp", {}) or {}
        push = raw.get("push", {}) or {}
        sens = raw.get("sens", {}) or {}
        success = raw.get("success", {}) or {}
        return cls(
            type=str(raw.get("type", cls.type)),
            t_obj=str(objects.get("t_obj", "t_obj")),
            t_target=str(objects.get("t_target", "t_target")),
            ee_site=str(raw.get("ee_site", "_tcp")),
            grasp_quat=tuple(float(v) for v in quat),
            gripper_cmd=_f(raw, "gripper_cmd", cls.gripper_cmd),
            window_width=_i(window, "width", cls.window_width),
            window_height=_i(window, "height", cls.window_height),
            workspace_x=_tup(ws, "x_range", cls.workspace_x),
            workspace_y=_tup(ws, "y_range", cls.workspace_y),
            tcp_z_min=_f(tcp, "z_min", cls.tcp_z_min),
            tcp_z_max=_f(tcp, "z_max", cls.tcp_z_max),
            tcp_initial_z=_f(tcp, "initial_z", cls.tcp_initial_z),
            push_z_min=_f(push, "z_min", cls.push_z_min),
            push_z_max=_f(push, "z_max", cls.push_z_max),
            sens_mouse=_f(sens, "mouse", cls.sens_mouse),
            sens_wheel=_f(sens, "wheel", cls.sens_wheel),
            sens_step=_f(sens, "step", cls.sens_step),
            sens_invert_x=bool(sens.get("invert_x", cls.sens_invert_x)),
            sens_invert_y=bool(sens.get("invert_y", cls.sens_invert_y)),
            success_dist=_f(success, "dist", cls.success_dist),
            success_yaw=_f(success, "yaw", cls.success_yaw),
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
