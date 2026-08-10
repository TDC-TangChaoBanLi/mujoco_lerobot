"""mujoco_lerobot — MuJoCo-based LeRobot simulation core.

Submodules:
- configs: YAML config loading (dataset / scene / task / teacher / sim)
- simulate: low-level MuJoCo wrapper, IK solver, camera renderer
- data: observation collection, controllers, scripted teachers, dataset writer

LeRobot 环境插件（gymnasium 环境 + EnvConfig 注册）已迁移至独立的
`lerobot_env_mujoco_lerobot` 包。
"""

__version__ = "0.1.0"

# 必须在任何 `import mujoco` 之前设置无头渲染后端（EGL），否则 glfw 后端已加载、
# 无 DISPLAY 时创建 mujoco.Renderer 会崩溃。
from .gl_setup import configure as _configure_headless_gl

_configure_headless_gl()
