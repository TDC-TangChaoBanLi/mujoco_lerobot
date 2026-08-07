"""lerobot 深度单位自动跟随补丁。

问题背景
--------
lerobot 训练时解码深度的单位由 ``dataset.depth_output_unit`` 决定，默认是 ``"mm"``
（见 ``lerobot.configs.video.DEFAULT_DEPTH_UNIT``）；而数据集 ``info.json`` 的
``depth_unit`` 声明了**记录单位**（本项目为 ``"m"``，来自 MuJoCo 米制渲染，采集时
按米写入）。二者若不显式对齐，训练解码（毫米）与评估 env 产出（米）的输入分布会
不一致，导致深度通道在评估时退化。

本补丁在 ``lerobot.datasets.factory.make_dataset`` 上打 monkeypatch：创建训练数据集
时，若数据集声明了 ``depth_unit``，自动用它作为解码输出单位（替代 lerobot 默认的
``"mm"``），使训练解码与评估 env 一致，**无需显式指定** ``dataset.depth_output_unit``。

行为
----
- 数据集有深度特征且记录了 ``depth_unit``：解码单位 = 记录单位（覆盖配置值，并打日志）。
- 数据集无深度 / 无记录单位：回退到配置的 ``depth_output_unit``。
- 通过 ``lerobot_env_mujoco_lerobot`` 插件导入自动安装；重复安装幂等。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

try:
    from lerobot.datasets import factory as _lerobot_factory
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
except Exception as _exc:  # pragma: no cover
    _lerobot_factory = None
    _LeRobotDatasetMetadata = None
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None

_PATCH_ATTR = "_lerobot_depth_unit_patched"


def _read_recorded_depth_unit_from_info(cfg) -> str | None:
    """直接从本地数据集 meta/info.json 读取深度记录单位（快速，不加载完整元数据）。"""
    root = getattr(cfg.dataset, "root", None)
    if root is None:
        return None
    info_path = Path(root) / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
    except Exception:
        return None
    units: set[str] = set()
    for ft in (info.get("features") or {}).values():
        finfo = ft.get("info") or {}
        if finfo.get("is_depth_map") and finfo.get("depth_unit"):
            units.add(str(finfo["depth_unit"]))
    return _dedupe_units(units)


def _resolve_recorded_depth_unit(cfg) -> str | None:
    """解析数据集记录的深度单位；无深度或无记录单位返回 None。"""
    unit = _read_recorded_depth_unit_from_info(cfg)
    if unit is not None:
        return unit
    if _LeRobotDatasetMetadata is None:
        return None
    # 回退：通过元数据（处理远端 / 非标准目录结构）
    ds_meta = _LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
        repo_type=cfg.dataset.repo_type,
    )
    units: set[str] = set()
    for key in ds_meta.depth_keys:
        unit = (ds_meta.features[key].get("info") or {}).get("depth_unit")
        if unit is not None:
            units.add(str(unit))
    return _dedupe_units(units)


def _dedupe_units(units: set[str]) -> str | None:
    if not units:
        return None
    if len(units) > 1:
        log.warning("数据集深度特征记录单位不一致: %s，将使用 %r", sorted(units), sorted(units)[0])
    return sorted(units)[0]


def _make_patched_make_dataset(original):
    def make_dataset(cfg):
        recorded = _resolve_recorded_depth_unit(cfg)
        if recorded is not None and recorded != cfg.dataset.depth_output_unit:
            log.info(
                "深度单位自动跟随数据集记录单位 depth_unit=%r（覆盖配置 "
                "depth_output_unit=%r）；如需强制覆盖请显式设置 "
                "--dataset.depth_output_unit=<m|mm>",
                recorded,
                cfg.dataset.depth_output_unit,
            )
            cfg.dataset.depth_output_unit = recorded
        return original(cfg)

    return make_dataset


def install_lerobot_depth_unit_patch() -> bool:
    """在 ``lerobot.datasets.factory.make_dataset`` 上安装补丁（幂等）。

    Returns:
        是否成功安装（lerobot 不可用或已被其他机制改动时返回 False）。
    """
    if _lerobot_factory is None:
        log.warning("无法导入 lerobot.datasets.factory，跳过深度单位自动跟随补丁: %s", _IMPORT_ERROR)
        return False
    if getattr(_lerobot_factory.make_dataset, _PATCH_ATTR, False):
        return True
    _lerobot_factory.make_dataset = _make_patched_make_dataset(_lerobot_factory.make_dataset)
    setattr(_lerobot_factory.make_dataset, _PATCH_ATTR, True)
    return True
