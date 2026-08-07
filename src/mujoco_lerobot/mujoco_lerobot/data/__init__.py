"""数据采集模块。

提供自动（scripted teacher）与键盘遥操作两条采集路径。
"""

from .reset_manager import ResetManager
from .observation_collector import ObservationCollector
from .controllers import Controller, ScriptedTeacherController, KeyboardTeleopController
from .dataset_writer import LeRobotDatasetWriter, LeRobotDatasetConfig
from .simulation_manager import SimulationManager

__all__ = [
    "ResetManager",
    "ObservationCollector",
    "Controller",
    "ScriptedTeacherController",
    "KeyboardTeleopController",
    "LeRobotDatasetWriter",
    "LeRobotDatasetConfig",
    "SimulationManager",
]
