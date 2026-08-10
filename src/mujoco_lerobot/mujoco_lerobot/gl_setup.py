"""无头（无 DISPLAY）时的 OpenGL 后端选择。

必须在**任何** `import mujoco` 之前调用：MuJoCo 的 `gl_context` 模块在首次导入时
就读取 `MUJOCO_GL` 环境变量并选定后端（Linux 下默认 glfw，需要真实 DISPLAY）。
若在无头环境（如 SSH 服务器）下已加载 glfw 后才设置 `MUJOCO_GL=egl`，则不会生效，
创建 `mujoco.Renderer` 时会报：

    FatalError: an OpenGL platform library has not been loaded into this process

本函数在 DISPLAY 缺失时把 `MUJOCO_GL` 设为 `egl`（离屏渲染，需 GPU + EGL 驱动，
如 NVIDIA），否则保持原值。使用 `setdefault` 尊重用户显式指定的后端。
"""

from __future__ import annotations

import os


def configure() -> None:
    """无头环境时把 MuJoCo 渲染后端切到 EGL（离屏）。幂等，可重复调用。"""
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
