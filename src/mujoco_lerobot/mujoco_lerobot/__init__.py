"""mujoco_lerobot — MuJoCo-based LeRobot simulation core.

Submodules:
- configs: YAML config loading (dataset / scene / task / teacher / sim)
- simulate: low-level MuJoCo wrapper, IK solver, camera renderer
- data: observation collection, controllers, scripted teachers, dataset writer

LeRobot 环境插件（gymnasium 环境 + EnvConfig 注册）已迁移至独立的
`lerobot_env_mujoco_lerobot` 包。
"""

__version__ = "0.1.0"
