"""项目路径解析。

所有 YAML 配置中的相对路径均以项目根目录为基准，
这里通过向上查找「同时包含 pyproject.toml 与 configs/」的目录来定位项目根。
"""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """从任意位置向上查找项目根目录（含 pyproject.toml 与 configs/ 的目录）。"""
    cur = (start or Path(__file__).resolve().parent).resolve()
    for _ in range(12):
        if (cur / "pyproject.toml").is_file() and (cur / "configs").is_dir():
            return cur
        cur = cur.parent
    # 兜底：当前工作目录
    return Path.cwd()


PROJECT_ROOT = find_project_root()
CONFIG_ROOT = PROJECT_ROOT / "configs"
ASSETS_ROOT = PROJECT_ROOT / "assets"
