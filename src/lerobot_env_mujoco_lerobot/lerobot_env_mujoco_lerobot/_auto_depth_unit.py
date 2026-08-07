"""训练深度单位自动识别补丁。

lerobot 的 ``dataset.depth_output_unit``（训练解码单位）默认是 ``"mm"``，与数据集
声明的记录单位（``info.json`` 的 ``depth_unit``，本项目为 ``"m"``，采集自 MuJoCo
米制渲染）相互独立。本模块在训练启动时 patch
``lerobot.datasets.factory.make_dataset``：当 ``depth_output_unit`` 保持默认
``"mm"``（即未显式指定）时，自动改为数据集记录的单位，使训练解码/stats 与评估
env（米）及数据集声明一致，训练侧无需再显式指定单位。

权衡：
- 这是对 lerobot 内部函数的 monkeypatch，lerobot 升级时可能需跟进维护。
- 显式 ``--dataset.depth_output_unit=mm`` 与默认值同为 ``"mm"``，无法区分，因此
  也会被记录单位覆盖。本项目始终记录米（env 产出米、MuJoCo 渲染米），跟随记录
  单位总是正确。
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_LEROBOT_DEFAULT_DEPTH_UNIT = "mm"

_installed = False


def _recorded_depth_unit(ds_meta) -> str | None:
    """读取数据集记录的深度单位（info.json 的 depth_unit），无深度特征时返回 None。"""
    for key in getattr(ds_meta, "depth_keys", []):
        info = (getattr(ds_meta, "features", {}).get(key) or {}).get("info") or {}
        unit = info.get("depth_unit")
        if unit is not None:
            return unit
    return None


def resolve_depth_output_unit(depth_output_unit: str, ds_meta) -> str:
    """解析应使用的 depth_output_unit：默认 "mm"（未显式指定）时跟随记录单位。"""
    if depth_output_unit != _LEROBOT_DEFAULT_DEPTH_UNIT:
        return depth_output_unit
    unit = _recorded_depth_unit(ds_meta)
    return unit if unit is not None else depth_output_unit


def install_auto_depth_output_unit() -> bool:
    """patch ``lerobot.datasets.factory.make_dataset``，返回是否安装成功（幂等）。"""
    global _installed
    if _installed:
        return True
    try:
        from lerobot.datasets import factory
    except Exception as exc:  # pragma: no cover
        log.warning("无法导入 lerobot.datasets.factory，跳过深度单位自动识别: %s", exc)
        return False

    original = factory.make_dataset

    def _make_dataset(cfg):
        try:
            dc = getattr(cfg, "dataset", None)
            if dc is not None:
                from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

                ds_meta = LeRobotDatasetMetadata(
                    dc.repo_id,
                    root=getattr(dc, "root", None),
                    revision=getattr(dc, "revision", None),
                    repo_type=getattr(dc, "repo_type", None),
                )
                resolved = resolve_depth_output_unit(
                    getattr(dc, "depth_output_unit", _LEROBOT_DEFAULT_DEPTH_UNIT), ds_meta
                )
                if resolved != getattr(dc, "depth_output_unit", None):
                    log.info(
                        "自动识别深度单位：数据集记录 depth_unit=%r → "
                        "dataset.depth_output_unit=%r（无需显式指定）",
                        _recorded_depth_unit(ds_meta), resolved,
                    )
                    dc.depth_output_unit = resolved
        except Exception as exc:  # pragma: no cover
            log.warning("自动识别深度单位失败，回退到默认: %s", exc)
        return original(cfg)

    factory.make_dataset = _make_dataset
    _installed = True
    log.info("已安装训练深度单位自动识别补丁（跟随数据集 depth_unit）")
    return True
