"""mujoco_lerobot — MuJoCo-based LeRobot simulation environment.

Submodules:
- configs: YAML config loading (dataset / scene / task / teacher / sim)
- simulate: low-level MuJoCo wrapper, IK solver, camera renderer
- data: observation collection, controllers, scripted teachers, dataset writer
- env: gymnasium environment + LeRobot EnvConfig plugin
"""

__version__ = "0.1.0"
