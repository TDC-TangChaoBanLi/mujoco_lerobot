# AGENTS.md — mujoco-lerobot

基于 MuJoCo 的 LeRobot 仿真环境：**数据采集 → 策略训练 → 评估**。全部 YAML 配置驱动、任务无关：任何新任务只需新增 teacher + 配置，不需要改动框架。

## 设计原则（务必遵守）

1. **任务无关框架**：`configs/`（配置加载）、`simulate/`（MuJoCo 封装 / IK / 渲染）、`data/`（采集链路）均为通用组件，**不含任何特定任务的逻辑**——包括"如果任务是 X 就……"的分支。
2. **特定任务代码只允许放在 `src/mujoco_lerobot/mujoco_lerobot/data/teachers/`**：teacher（脚本式/遥操作，含任务专属状态机、成功判定、录制决策）+ 任务专属 YAML（`configs/tasks/`、`configs/scenes/`、`configs/teachers/`、`configs/dataset/`）。外部任务包也可自行注册 teacher。
3. **框架不满足新任务需求时，先对框架做兼容性变更**（通用化、可配置化），而不是在框架里加任务特化代码。变更须保持向后兼容（默认值与旧行为一致），并用测试固定。
4. **接入点靠注册表与配置连接，而非代码分支**：teacher 用 `@register_teacher("XxxTeacher")` 注册（含 `config_class`），任务用 `configs/tasks/tasks.yaml` 连接到 scene / teacher 配置；采集脚本、评估环境、训练链路均无任务分支。

### 新增任务的流程
1. **场景**：`assets/mujoco/scenes/<task>.xml` + `configs/scenes/scene_<task>.yaml`（机器人/相机/视角/`ik_solver`）。
2. **teacher**：在 `data/teachers/<task>_teacher.py` 继承 `Teacher`，`@register_teacher` 注册，实现 `step()`（输出各臂 `[x,y,z,qw,qx,qy,qz,gripper]` 目标位姿）与 `check_success()`（基于物理状态的评估判定）；遥操作类覆盖 `recording_decision()` 与 `start_collection()/end_collection()` 钩子。配 `configs/teachers/<task>.yaml`。
3. **连接**：`configs/tasks/tasks.yaml` 注册任务（scene_file + scene_config_file + teacher_config_file + domain_randomization）。
4. **采集/训练/评估** 直接用下方通用命令（`--task <task>`）。

## 架构 / 包布局

- `src/mujoco_lerobot/`（core 包 `mujoco_lerobot`）
  - `configs/`：YAML 配置加载（tasks / scenes / dataset / teachers / simulate）——`RobotConfig.ik_solver` 为 IK 参数
  - `simulate/`：`MujocoWrapper`（MuJoCo 封装）、`MinkIK`（mink IK）、`CameraRenderer`（多相机并行渲染，深度米制）——通用仿真组件
  - `data/`：采集链路（`ObservationCollector` / `SimulationManager` / `ScriptedTeacherController` / `KeyboardTeleopController` / `ResetManager` / `LeRobotDatasetWriter`）；**录制生命周期由 teacher 控制**（`run_episode` 每策略步询问 `controller.recording_decision`，返回 START/SAVED/DISCARDED/QUIT）；**采集会话钩子** `start_collection()/end_collection()` + `retry_limit`
  - `data/teachers/`：**任务专属代码唯一允许的位置**（pick_place / dual_pick_place / push_t + push_t 遥操作 `push_t_teleop.py`）
- `src/lerobot_env_mujoco_lerobot/`：LeRobot 评估环境插件（gym env，type=`mujoco_lerobot`；深度单位显式米）
- `src/lerobot_policy_Adaptive_ACT/`：自适应 ACT 策略插件（type=`adaptive_act`）
- `libs/mujoco_camrender/`：多相机并行渲染（C++ + pybind，git submodule）
- `configs/`：tasks / scenes / dataset / teachers / policy
- `scripts/collect_data.py`：一键采集（auto scripted teacher / keyboard / mouse / --append 追加）
- `tests/`：87 项测试（含无头遥操作测试）

## 端到端工作流

### 采集
```bash
# auto（scripted teacher）
uv run python scripts/collect_data.py --task-config configs/tasks/tasks.yaml \
    --task pick_place --dataset-config configs/dataset/dataset_pick_place.yaml \
    --episodes 50 [--no-render]

# PushT 鼠标遥操作（push_t 任务 teacher 模式下自动启动 pygame 2D 遥操作窗口，需 DISPLAY）
uv run python scripts/collect_data.py --task-config configs/tasks/tasks.yaml \
    --task push_t --dataset-config configs/dataset/dataset_push_t.yaml --episodes 20

# 追加采集（--episodes 为本轮新增条数；须配置一致性检查：fps/features/depth_unit/crf 硬性一致）
uv run python scripts/collect_data.py ... --append outputs/datasets/pick_place/<已有目录>
```
- 无头必须 `--no-render`（WSL/D3D12 下 `mujoco.viewer` 会 SIGSEGV）
- 输出 `outputs/datasets/<task>/<时间戳>/`（rgb+depth 视频编码，深度默认无损 HEVC）

#### PushT 鼠标遥操作（push_t 任务 teacher 模式自动启动）
- 按键：左键=切换鼠标控制 / 右键=开始·结束并保存 / 中键=丢弃当前集 / 滚轮=TCP 升降 / `[ ]`=鼠标灵敏度 / `- =`=滚轮灵敏度 / Esc=退出
- 窗口为 2D 俯视投影（pygame）：桌面范围、T 物块（绿，按 yaw 旋转）、目标 T（蓝）、当前 TCP（白点）、期望 TCP（黄十字）、状态栏（REC / MOUSE ON/OFF / TCP 高度 / PUSH OK-NO / 灵敏度）
- 配置：`configs/teachers/push_t.yaml`（window/tcp/push/sens/success）+ `scene_push_t.yaml` default_qpos（初始 TCP 落可推动带）+ `tasks.yaml`（t_obj_mount 随机化）
- 架构：`TeleopState`（线程安全，无 pygame 依赖，可无头测试）由窗口线程写、仿真线程读；teacher 经 `teacher_kwargs={"state": state}` 注入共享状态；`recording_decision` 每策略步先 `publish_teleop_state` 回写物理状态再消费鼠标事件；窗口生命周期由 `start_collection`/`end_collection` 钩子管理（`retry_limit=1`）

### 训练
```bash
uv run lerobot-train --config_path=configs/policy/adaptive_act.yaml \
    --dataset.root=outputs/datasets/pick_place/<时间戳> --steps=100000
```
- 纯 RGB：`input_features` 只选 rgb；RGBD：rgb+depth 用 `concat_visual_features` 拼 4 通道（conv1 自动适配）
- 深度单位两端显式米（记录侧 features info + 训练侧 `dataset.depth_output_unit: m`），不依赖补丁

### 评估
```bash
uv run lerobot-eval --env.type=mujoco_lerobot --env.task=pick_place \
    --env.dataset_config=configs/dataset/dataset_pick_place.yaml \
    --policy.path=<实际训练 output_dir>/checkpoints/last/pretrained_model \
    --eval.n_episodes=5 --env.max_episode_steps=4000
```
- `--policy.path` 必须匹配实际训练的 `--output_dir`（路径错会报 `ParsingError: Couldn't instantiate class EvalPipelineConfig`）
- 给足 `max_episode_steps=4000`（约 40s）

### 测试
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/
```
（`test_depth_unit_explicitly_unified_as_m` 需 HF 网络，离线环境跳过）

## 关键配置

- `configs/tasks/tasks.yaml`：任务 → scene / teacher 配置连接（含 domain_randomization）
- `configs/scenes/scene_*.yaml`：机器人 / 相机 / 视角 / **`robot[].ik_solver`**（mink IK 参数，见下）
- `configs/dataset/dataset_*.yaml`：记录格式 + `state.camera.depth_range` + `depth_crf`/`rgb_crf`
- `configs/teachers/*.yaml`：scripted teacher 参数（push_t 为遥操作参数）
- `configs/policy/adaptive_act.yaml`：训练配置（含 RGBD concat 示例）

### 机器人 ik_solver 参数（mink IK）
`configs/scenes/scene_*.yaml → robot[].ik_solver`，全部 IK 参数在此配置：
- `vel_limit`：关节角速度限制 (rad/s)，缺省 `[3.1416]*6`
- `pos_cost` / `ori_cost` / `gain` / `lm_damping`：FrameTask 参数（缺省 1.0 / 1.0 / 1.0 / 1e-6）
- `posture_cost`：PostureTask 正则（缺省 1e-3）
- `solver` / `damping` / `safety_break`：solve_ik 参数（缺省 daqp / 1e-12 / false）
- `max_iters` / `pos_threshold` / `ori_threshold`：多迭代收敛（缺省 1 / 0.01 / 0.1；`max_iters=1` 等价单次求解，`>1` 时按阈值提前退出，同官方 arm_ur5e_actuators.py）
- `collision_avoidance`：`enabled`（缺省 false）+ `gain`/`minimum_distance`/`detection_distance`/`bound_relaxation`/`broadphase` + `pairs`。pairs 写简短名（如 `ur_wrist_3_link`）自动匹配 `{prefix}COLLISION_{name}*`（回退 `{prefix}{name}*`、精确名），支持显式对侧臂名（`B_ur_wrist_3_link`）；**pairs 中不得含被抓取物体**（如 `cube_geom`），否则 IK 拒绝接近

## 重要坑 / 约定

1. **归一化一致性**：rgb 训练=[0,1]↔评估 env `/255`；depth 训练=米（`depth_output_unit=m`）↔评估 env 米。不一致时策略"失明"、输出退化动作
2. **深度单位**：两端显式统一为米，不依赖 monkeypatch（lerobot 升级无需维护补丁）
3. **评估 seed**：`env.reset(seed)` 真正控制物体随机化；`lerobot-eval` 每 episode 用 `seed+idx` 播种 → 可复现且各 episode 不同
4. **深度压缩**：默认无损 HEVC（文件约 rgb 18 倍）；设 `depth_crf` 走有损（须 `extra_options={}` 去掉 `lossless=1`）
5. **eval 视频帧率** = recode_hz；评估时长给足 `max_episode_steps=4000`
6. **续训**：`lerobot_train --config_path=<ckpt>/pretrained_model --resume=true --output_dir=<原目录> --steps=<新总步数>`
7. **插件机制**：包名以 `lerobot_policy_` / `lerobot_env_` 开头自动导入注册（`@PreTrainedConfig.register_subclass` / `@EnvConfig.register_subclass`）
8. **训练输入特征**：rgb(3ch) 与 depth(1ch) 两个 VISUAL 都选且不拼接会报 "same number of channels"；只选其一或 concat 成 RGBD
9. **PushT 遥操作**：push_t 任务 teacher 模式下由 `PushTTeacher.start_collection` 自动启动 pygame 2D 遥操作窗口（需 DISPLAY）；`--no-render` 只关 3D viewer 不影响 2D 窗口。录制由 `recording_decision` 控制（右键 START/SAVED、中键 DISCARDED、Esc QUIT；`retry_limit=1`，discarded 为用户主动丢弃不重试）。freejoint 在 `t_obj_mount`（domain_randomization key 用 mount 名）。窗口 UI 文本必须英文（中文乱码；终端 print 中文正常）
10. **PushT default_qpos**：`scene_push_t.yaml` default_qpos 使 TCP 初始落点 (0.34, -0.13, 0.69) 在可推动带 [0.66, 0.72] 内，开局即可推 T；改关节角后须用 IK 反解验证可达性与高度
11. **PushT 鼠标方向反转**：本环境 grab 模式（MOUSE ON）下 SDL 真实鼠标增量与物理移动 180° 相反（X/Y 都反），XTest/warp/非 grab 均标准 → 无法自动校准，已用配置 `sens.invert_x/y`（当前 true）修正，标准环境改回 false
12. **camrender 重编译**：`libs/mujoco_camrender` 任何 C++/pybind 改动后必须重编译 `.so` 并 `uv sync --reinstall-package mujoco-camrender`，否则运行时枚举/API 不匹配（如 `RenderBackend.EGL` 缺失 → 回退 mujoco.Renderer）
13. **位置控制器带宽**：position actuator 是 PD（`τ=kp(ctrl−q)−kv·q̇`，dampratio=1 临界阻尼，`kv=2ζ√(kp·m)`）。肩肘带宽仅 ~1 Hz（kp 200/100 vs 惯量 2–3.5 kg·m²），每 policy 步仅完成 ~8% 误差衰减 → 物理速度 ≈ vel_limit × ~0.1 跟踪系数；**vel_limit 是 IK 指令层配额，与物理层 PD 带宽互相独立**。想加快应提高 kp（注意数值稳定性），不是调 vel_limit

## 当前状态

- 场景：pick_place（2 相机）/ dual_pick_place（3 相机）/ push_t
- 已训练 RGBD adaptive_act：`outputs/train/rgbd_adaptive_act_pick_place/`（100K 步，loss ~0.022）
- 数据集：`outputs/datasets/{pick_place,push_t}/<时间戳>/`（rgb+depth，米制；pick_place 含 50-episode 数据集 `20260806_004855`）
