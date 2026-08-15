"""Scripted Teacher 专家模块。

提供 teacher 注册表与自动发现机制：
  - 任何继承 Teacher 的子类，用 `@register_teacher("XxxTeacher")` 装饰后即注册，
    无需在此处手动登记，支持外部任务在各自模块中注册。
  - 注册时同时登记 `config_class`（teacher 配置 dataclass），供配置加载与实例化。
"""

from __future__ import annotations

from typing import Any

import mujoco

from .base import Teacher, TeacherState

# 类型 → 类 注册表（由 controllers / env 按 teacher 配置的 type 实例化）
TEACHER_REGISTRY: dict[str, type[Teacher]] = {}

# 类型 → 配置类 注册表（由 load_teacher_config 按 type 加载对应配置）
TEACHER_CONFIG_REGISTRY: dict[str, type] = {}


def register_teacher(teacher_type: str | None = None):
    """装饰器：将 teacher 子类注册到 TEACHER_REGISTRY。

    用法:
        @register_teacher("PushTTeacher")
        class PushTTeacher(Teacher):
            teacher_type = "PushTTeacher"
            config_class = PushTTeacherConfig
            ...

    不传 teacher_type 时使用类的 `teacher_type` 属性。
    同时会把 `config_class` 登记到 TEACHER_CONFIG_REGISTRY。
    """

    def decorator(cls):
        ttype = teacher_type or getattr(cls, "teacher_type", None)
        if not ttype:
            raise ValueError(
                f"teacher {cls.__name__} 必须提供 teacher_type 或 @register_teacher 参数"
            )
        cls.teacher_type = ttype
        if ttype in TEACHER_REGISTRY:
            raise ValueError(f"teacher 类型重复注册: {ttype!r}")
        TEACHER_REGISTRY[ttype] = cls
        cfg_cls = getattr(cls, "config_class", None)
        if cfg_cls is not None:
            TEACHER_CONFIG_REGISTRY[ttype] = cfg_cls
        return cls

    return decorator


def create_teacher(
    teacher_type: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: Any | None = None,
    prefix: str = "",
    **kwargs: Any,
) -> Teacher:
    """按 teacher 类型实例化 teacher（自动发现注册的子类）。

    ``**kwargs``（如 ``state=TeleopState``）透传给 teacher 构造，
    供遥操作类 teacher 注入共享状态。
    """
    cls = TEACHER_REGISTRY.get(teacher_type)
    if cls is None:
        raise KeyError(f"未知 teacher 类型: {teacher_type!r}。可用: {sorted(TEACHER_REGISTRY)}")
    return cls(model, data, config=config, prefix=prefix, **kwargs)


def get_teacher_config_class(teacher_type: str) -> type:
    """返回 teacher 类型对应的配置 dataclass。"""
    cls = TEACHER_CONFIG_REGISTRY.get(teacher_type)
    if cls is None:
        raise KeyError(
            f"teacher {teacher_type!r} 未注册 config_class，无法加载配置。"
            f"可用: {sorted(TEACHER_CONFIG_REGISTRY)}"
        )
    return cls


def discover_teachers() -> dict[str, type[Teacher]]:
    """触发全部已知 teacher 模块导入以完成注册，并返回注册表。

    新增任务时若在其它包中注册 teacher，可在此处或入口处 import 对应模块。
    """
    # 内置 teacher 模块（import 即触发 register_teacher）
    from . import (  # noqa: F401
        pick_place_teacher,
        dual_pick_place_teacher,
        push_t_teacher,
    )

    return TEACHER_REGISTRY


def load_teacher_config_for_type(teacher_type: str, path: str) -> Any:
    """加载指定类型 teacher 的配置（与 configs.teacher_config 联动）。"""
    from ...configs.teacher_config import load_teacher_config

    return load_teacher_config(path)


# 导入内置 teacher 模块完成注册
from . import (  # noqa: E402,F401
    pick_place_teacher,
    dual_pick_place_teacher,
    push_t_teacher,
)
from .pick_place_teacher import PickPlaceTeacher, PickPlaceState  # noqa: E402
from .dual_pick_place_teacher import DualPickPlaceTeacher, DualPickPhase  # noqa: E402
from .push_t_teacher import PushTTeacher  # noqa: E402

__all__ = [
    "Teacher",
    "TeacherState",
    "PickPlaceTeacher",
    "PickPlaceState",
    "DualPickPlaceTeacher",
    "DualPickPhase",
    "PushTTeacher",
    "TEACHER_REGISTRY",
    "TEACHER_CONFIG_REGISTRY",
    "register_teacher",
    "create_teacher",
    "get_teacher_config_class",
    "discover_teachers",
]
