"""配置加载 — 从 YAML 加载仿真 / 场景 / 任务配置。

层次关系（去掉了原项目的硬编码 _SCENE_FILE_TO_YAML 映射，改为显式连接）:
    simulate_default.yaml          → 仿真与采集通用参数
    scenes/scene_*.yaml            → 场景（机器人、相机）
    tasks/tasks.yaml               → 任务（scene_file + scene_config_file +
                                     teacher_config_file + domain_randomization）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .paths import CONFIG_ROOT, PROJECT_ROOT

log = logging.getLogger(__name__)

_DEFAULT_TASK_FILE = "configs/tasks/tasks.yaml"
_SIM_DEFAULT_FILE = "configs/simulate_default.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def resolve_config_path(path: str | Path) -> Path:
    """将相对路径解析为项目根下的绝对路径（绝对路径原样返回）。"""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _range_tuple(d: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    v = d.get(key)
    if v is None:
        return default
    return (float(v[0]), float(v[1]))


# ═══════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════


@dataclass
class SimParams:
    physics_dt: float = 0.001
    policy_dt: float = 0.01

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimParams":
        return cls(
            physics_dt=float(d.get("physics_dt", cls.physics_dt)),
            policy_dt=float(d.get("policy_dt", cls.policy_dt)),
        )

    @property
    def policy_steps(self) -> int:
        """每个 policy 步内的物理步数。"""
        return max(1, round(self.policy_dt / self.physics_dt))


@dataclass
class CollectionParams:
    max_time: float = 30.0
    max_attempts: int = 3

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CollectionParams":
        return cls(
            max_time=float(d.get("max_time", cls.max_time)),
            max_attempts=int(d.get("max_attempts", cls.max_attempts)),
        )


@dataclass
class CollisionAvoidanceConfig:
    """mink CollisionAvoidanceLimit 配置（默认关闭，场景按需开启）。

    参数含义见 mink.CollisionAvoidanceLimit：
      - pairs: list[list[str]] 每组两个 geom 名称列表（简短名自动匹配
        `{prefix}COLLISION_{name}*`，支持显式写完整 geom 名或对侧臂名）。
    """

    enabled: bool = False
    gain: float = 0.85
    minimum_distance: float = 0.005
    detection_distance: float = 0.01
    bound_relaxation: float = 0.0
    broadphase: bool = True
    pairs: list[tuple[list[str], list[str]]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CollisionAvoidanceConfig":
        if not d:
            return cls()
        raw_pairs = d.get("pairs", []) or []
        pairs: list[tuple[list[str], list[str]]] = []
        for p in raw_pairs:
            if len(p) != 2:
                raise ValueError(f"collision_avoidance.pairs 每项须为 [geomA列表, geomB列表]: {p}")
            pairs.append(([str(g) for g in p[0]], [str(g) for g in p[1]]))
        return cls(
            enabled=bool(d.get("enabled", False)),
            gain=float(d.get("gain", cls.gain)),
            minimum_distance=float(d.get("minimum_distance", cls.minimum_distance)),
            detection_distance=float(d.get("detection_distance", cls.detection_distance)),
            bound_relaxation=float(d.get("bound_relaxation", cls.bound_relaxation)),
            broadphase=bool(d.get("broadphase", True)),
            pairs=pairs,
        )


@dataclass
class IKSolverConfig:
    """mink 逆运动学求解器参数（`robot/ik_solver` 子块）。

    默认值与 MinkIK 原有硬编码一致，保证未配置时行为不变。
    vel_limit 缺省为 [3.1416]*6（UR 类关节典型限速）。
    """

    vel_limit: list[float] = field(
        default_factory=lambda: [3.1416] * 6
    )
    # FrameTask
    pos_cost: float = 1.0
    ori_cost: float = 1.0
    gain: float = 1.0
    lm_damping: float = 1e-6
    # PostureTask
    posture_cost: float = 1e-3
    # solve_ik
    solver: str = "daqp"
    damping: float = 1e-12
    safety_break: bool = False
    # 多迭代收敛（max_iters=1 等价于单次求解）
    max_iters: int = 1
    pos_threshold: float = 0.01
    ori_threshold: float = 0.1
    # 碰撞避免
    collision_avoidance: CollisionAvoidanceConfig = field(
        default_factory=CollisionAvoidanceConfig
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "IKSolverConfig":
        if not d:
            return cls()
        return cls(
            vel_limit=[float(v) for v in d.get("vel_limit", [3.1416] * 6)],
            pos_cost=float(d.get("pos_cost", cls.pos_cost)),
            ori_cost=float(d.get("ori_cost", cls.ori_cost)),
            gain=float(d.get("gain", cls.gain)),
            lm_damping=float(d.get("lm_damping", cls.lm_damping)),
            posture_cost=float(d.get("posture_cost", cls.posture_cost)),
            solver=str(d.get("solver", cls.solver)),
            damping=float(d.get("damping", cls.damping)),
            safety_break=bool(d.get("safety_break", False)),
            max_iters=int(d.get("max_iters", 1)),
            pos_threshold=float(d.get("pos_threshold", cls.pos_threshold)),
            ori_threshold=float(d.get("ori_threshold", cls.ori_threshold)),
            collision_avoidance=CollisionAvoidanceConfig.from_dict(
                d.get("collision_avoidance")
            ),
        )


@dataclass
class RobotConfig:
    """场景中一个机械臂实例的配置。"""

    name: str
    prefix: str = ""
    arm_joints: list[str] = field(default_factory=list)
    gripper_joints: list[str] = field(default_factory=list)
    ee_site: str = "_tcp"
    default_qpos: list[float] = field(default_factory=list)
    ik_solver: IKSolverConfig = field(default_factory=IKSolverConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RobotConfig":
        return cls(
            name=str(d.get("name", "")),
            prefix=str(d.get("prefix", "")),
            arm_joints=[str(j) for j in d.get("arm_joints", [])],
            gripper_joints=[str(j) for j in d.get("gripper_joints", [])],
            ee_site=str(d.get("ee_site", "_tcp")),
            default_qpos=[float(v) for v in d.get("default_qpos", [])],
            ik_solver=IKSolverConfig.from_dict(d.get("ik_solver")),
        )

    @property
    def prefixed_arm_joints(self) -> list[str]:
        return [f"{self.prefix}{j}" for j in self.arm_joints]

    @property
    def prefixed_gripper_joints(self) -> list[str]:
        return [f"{self.prefix}{j}" for j in self.gripper_joints]

    @property
    def prefixed_ee_site(self) -> str:
        return f"{self.prefix}{self.ee_site}"

    @property
    def n_arm_joints(self) -> int:
        return len(self.arm_joints)

    @property
    def n_gripper_joints(self) -> int:
        return len(self.gripper_joints)


@dataclass
class CameraConfig:
    """场景中一个相机的配置。image_size 为 (W, H)。

    注意：深度编码所需的 depth_range 不在此定义（它属于数据集配置，
    见 configs/dataset/dataset_*.yaml → state.camera.depth_range），
    仅 lerobot 深度图像编码器需要它。
    """

    name: str
    fps: int = 30
    image_size: tuple[int, int] = (640, 480)
    type: str = "rgb_depth"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CameraConfig":
        img = d.get("image_size", [640, 480])
        return cls(
            name=str(d["name"]),
            fps=int(d.get("fps", 30)),
            image_size=(int(img[0]), int(img[1])),
            type=str(d.get("type", "rgb_depth")),
        )

    @property
    def width(self) -> int:
        return self.image_size[0]

    @property
    def height(self) -> int:
        return self.image_size[1]

    @property
    def dt(self) -> float:
        return 1.0 / self.fps


@dataclass
class ObjectRandomization:
    """物体随机化参数（域随机化）。"""

    x_range: tuple[float, float] = (0.0, 0.0)
    y_range: tuple[float, float] = (0.0, 0.0)
    z_range: tuple[float, float] = (0.0, 0.0)
    roll_range: tuple[float, float] = (0.0, 0.0)
    pitch_range: tuple[float, float] = (0.0, 0.0)
    yaw_range: tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ObjectRandomization":
        pos = d.get("pos", {}) or {}
        euler = d.get("euler", {}) or {}
        return ObjectRandomization(
            x_range=_range_tuple(pos, "x_range", (0.0, 0.0)),
            y_range=_range_tuple(pos, "y_range", (0.0, 0.0)),
            z_range=_range_tuple(pos, "z_range", (0.0, 0.0)),
            roll_range=_range_tuple(euler, "roll_range", (0.0, 0.0)),
            pitch_range=_range_tuple(euler, "pitch_range", (0.0, 0.0)),
            yaw_range=_range_tuple(euler, "yaw_range", (0.0, 0.0)),
        )


@dataclass
class TaskConfig:
    name: str
    scene_file: str
    scene_config_file: str
    teacher_config_file: str
    task_id: int = 0
    objects: dict[str, ObjectRandomization] = field(default_factory=dict)

    @property
    def scene_path(self) -> Path:
        return resolve_config_path(self.scene_file)

    @property
    def teacher_config_path(self) -> Path:
        return resolve_config_path(self.teacher_config_file)

    @property
    def scene_config_path(self) -> Path:
        return resolve_config_path(self.scene_config_file)


@dataclass
class ViewConfig:
    """Viewer 视角参数（评估可视化 / 数据采集初始视角 / eval 视频渲染）。

    对应 MuJoCo free camera（MjvCamera）参数；采集时在仿真开始前应用到
    viewer，评估时用于录制 eval 视频。image_size 控制 eval 视频分辨率。
    """

    lookat: tuple[float, float, float] = (0.45, 0.0, 0.65)
    distance: float = 1.8
    elevation: float = -25.0
    azimuth: float = 130.0
    image_size: tuple[int, int] = (640, 480)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ViewConfig":
        if not d:
            return cls()
        lookat = d.get("lookat", [0.45, 0.0, 0.65])
        img = d.get("image_size", [640, 480])
        return cls(
            lookat=tuple(float(v) for v in lookat),
            distance=float(d.get("distance", 1.8)),
            elevation=float(d.get("elevation", -25.0)),
            azimuth=float(d.get("azimuth", 130.0)),
            image_size=(int(img[0]), int(img[1])),
        )

    @property
    def width(self) -> int:
        return self.image_size[0]

    @property
    def height(self) -> int:
        return self.image_size[1]


@dataclass
class SceneConfig:
    """完整仿真配置。"""

    task: TaskConfig
    sim: SimParams = field(default_factory=SimParams)
    collection: CollectionParams = field(default_factory=CollectionParams)
    robots: list[RobotConfig] = field(default_factory=list)
    cameras: list[CameraConfig] = field(default_factory=list)
    view: ViewConfig = field(default_factory=ViewConfig)

    @property
    def n_arms(self) -> int:
        return len(self.robots)

    @property
    def action_dim(self) -> int:
        return sum(r.n_arm_joints + r.n_gripper_joints for r in self.robots)

    @property
    def state_dim(self) -> int:
        return sum(r.n_arm_joints + r.n_gripper_joints for r in self.robots)

    def robot_by_prefix(self, prefix: str) -> RobotConfig:
        for r in self.robots:
            if r.prefix == prefix:
                return r
        raise KeyError(f"无 prefix={prefix!r} 的机器人")

    def camera_by_name(self, name: str) -> CameraConfig:
        for c in self.cameras:
            if c.name == name:
                return c
        raise KeyError(f"无相机 {name!r}")


# ═══════════════════════════════════════════════════════
# 加载入口
# ═══════════════════════════════════════════════════════


def load_sim_params(sim_file: str | Path = _SIM_DEFAULT_FILE) -> SimParams:
    raw = _load_yaml(resolve_config_path(sim_file))
    return SimParams.from_dict(raw.get("sim", {}))


def load_collection_params(sim_file: str | Path = _SIM_DEFAULT_FILE) -> CollectionParams:
    raw = _load_yaml(resolve_config_path(sim_file))
    return CollectionParams.from_dict(raw.get("collection", {}))


def load_tasks(task_config_file: str | Path = _DEFAULT_TASK_FILE) -> dict[str, TaskConfig]:
    """加载任务配置文件中的所有任务。"""
    path = resolve_config_path(task_config_file)
    if not path.exists():
        raise FileNotFoundError(f"任务配置文件不存在: {path}")
    raw = _load_yaml(path)
    tasks: dict[str, TaskConfig] = {}
    for name, cfg in raw.items():
        cfg = cfg or {}
        tasks[name] = TaskConfig(
            name=name,
            scene_file=str(cfg.get("scene_file", "")),
            scene_config_file=str(cfg.get("scene_config_file", "")),
            teacher_config_file=str(cfg.get("teacher_config_file", "")),
            task_id=int(cfg.get("task_id", 0)),
            objects={
                obj: ObjectRandomization.from_dict(oc)
                for obj, oc in (cfg.get("domain_randomization", {}) or {}).items()
            },
        )
    return tasks


def get_task_list(task_config_file: str | Path = _DEFAULT_TASK_FILE) -> list[str]:
    return sorted(load_tasks(task_config_file).keys())


def load_task_config(
    task_name: str, task_config_file: str | Path = _DEFAULT_TASK_FILE
) -> TaskConfig:
    tasks = load_tasks(task_config_file)
    if task_name not in tasks:
        raise KeyError(
            f"任务 {task_name!r} 未找到。可用任务: {sorted(tasks)}"
        )
    return tasks[task_name]


def _load_robots(raw_scene: dict[str, Any]) -> list[RobotConfig]:
    return [
        RobotConfig.from_dict(r) for r in raw_scene.get("robot", [])
    ]


def _load_cameras(raw_scene: dict[str, Any]) -> list[CameraConfig]:
    return [
        CameraConfig.from_dict(c) for c in raw_scene.get("camera", [])
    ]


def load_scene_config(
    task_name: str, task_config_file: str | Path = _DEFAULT_TASK_FILE
) -> SceneConfig:
    """加载完整场景配置 = sim_default + task + scene。"""
    task = load_task_config(task_name, task_config_file)
    if not task.scene_config_file:
        raise ValueError(f"任务 {task_name!r} 缺少 scene_config_file")

    scene_path = task.scene_config_path
    if not scene_path.exists():
        raise FileNotFoundError(f"场景配置文件不存在: {scene_path}")
    raw_scene = _load_yaml(scene_path)

    return SceneConfig(
        task=task,
        sim=load_sim_params(),
        collection=load_collection_params(),
        robots=_load_robots(raw_scene),
        cameras=_load_cameras(raw_scene),
        view=ViewConfig.from_dict(raw_scene.get("view")),
    )


# ═══════════════════════════════════════════════════════
# 与 mjcf 模型的冲突校验（打印警告）
# ═══════════════════════════════════════════════════════


def validate_config_vs_model(config: SceneConfig, model: Any) -> list[str]:
    """校验配置与已加载的 MuJoCo 模型的一致性，返回警告列表。

    检查项：
      - 相机名称存在于模型中
      - yaml 相机 image_size 与 mjcf cam_resolution 不一致（警告）
      - yaml 相机 fps 与 mjcf cam_fovy 不冲突（fovy 不参与比较，仅提示）
      - 机器人关节 / 末端 site 名称存在于模型中
    """
    warnings: list[str] = []
    task_name = config.task.name

    # ── 相机 ──
    mj_cam_names = {model.camera(i).name for i in range(model.ncam)}
    for cam in config.cameras:
        if cam.name not in mj_cam_names:
            warnings.append(
                f"[{task_name}] 场景配置中的相机 {cam.name!r} 不存在于 mjcf 模型 "
                f"(可用: {sorted(mj_cam_names)})"
            )
        else:
            cam_id = model.camera(cam.name).id
            mj_w = int(model.cam_resolution[cam_id, 0])
            mj_h = int(model.cam_resolution[cam_id, 1])
            if (cam.width, cam.height) != (mj_w, mj_h):
                warnings.append(
                    f"[{task_name}] 相机 {cam.name!r} 分辨率冲突: "
                    f"yaml={cam.width}x{cam.height} vs mjcf={mj_w}x{mj_h}，"
                    f"将以 yaml 为准渲染"
                )

    # ── 机器人 ──
    mj_joint_names = {model.joint(i).name for i in range(model.njnt)}
    mj_site_names = {model.site(i).name for i in range(model.nsite)}
    for r in config.robots:
        for j in r.prefixed_arm_joints + r.prefixed_gripper_joints:
            if j not in mj_joint_names:
                warnings.append(
                    f"[{task_name}] 机器人 {r.name!r} 的关节 {j!r} 不存在于 mjcf 模型"
                )
        ee = r.prefixed_ee_site
        if ee not in mj_site_names:
            warnings.append(
                f"[{task_name}] 机器人 {r.name!r} 的末端 site {ee!r} 不存在于 mjcf 模型"
            )

    for w in warnings:
        log.warning(w)
    return warnings
