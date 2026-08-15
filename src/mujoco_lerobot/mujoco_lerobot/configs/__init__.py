"""配置加载模块。

从 YAML 加载数据集 / 场景 / 任务 / teacher / 仿真配置。
"""

from .paths import PROJECT_ROOT, CONFIG_ROOT, ASSETS_ROOT
from .config_loader import (
    SimParams,
    CollectionParams,
    RobotConfig,
    CameraConfig,
    ObjectRandomization,
    TaskConfig,
    SceneConfig,
    load_sim_params,
    load_collection_params,
    load_tasks,
    get_task_list,
    load_task_config,
    load_scene_config,
    validate_config_vs_model,
    resolve_config_path,
)
from .dataset_config import (
    DataSource,
    DatasetConfig,
)
from .teacher_config import (
    TeacherThresh,
    TeacherHeights,
    GripperCmd,
    PickPlaceTeacherConfig,
    DualPickPlaceTeacherConfig,
    PushTTeacherConfig,
    load_teacher_config,
)

__all__ = [
    "PROJECT_ROOT", "CONFIG_ROOT", "ASSETS_ROOT",
    "SimParams", "CollectionParams", "RobotConfig", "CameraConfig",
    "ObjectRandomization", "TaskConfig", "SceneConfig",
    "load_sim_params", "load_collection_params", "load_tasks",
    "get_task_list", "load_task_config", "load_scene_config",
    "validate_config_vs_model", "resolve_config_path",
    "DataSource", "DatasetConfig",
    "TeacherThresh", "TeacherHeights", "GripperCmd",
    "PickPlaceTeacherConfig", "DualPickPlaceTeacherConfig",
    "PushTTeacherConfig",
    "load_teacher_config",
]
