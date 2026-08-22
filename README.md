# mujoco-lerobot

基于 MuJoCo 的 LeRobot 仿真环境：**数据采集 → 策略训练 → 策略评估** 全流程闭环。

- 自动（scripted teacher）/ 键盘遥操作数据采集，由 YAML 配置驱动，LeRobot 流式编码写入（内存恒定）
- 高性能多相机并行渲染（[mujoco_camrender](libs/mujoco_camrender)，RGB + 深度一次渲染），支持无头 EGL 离屏
- IK 解算基于 [mink](https://github.com/kevinzakka/mink/)
- LeRobot 环境插件（`lerobot_env_mujoco_lerobot`），`lerobot-eval` 一键评估
- 自适应 ACT 策略插件（`lerobot_policy_Adaptive_ACT`）：任意图像通道数（灰度 / RGB / RGB-D）、
  按相机分组共享 / 独立 resnet18 backbone，注册为策略类型 `adaptive_act`
- 深度单位全程显式统一为**米**（记录 `depth_unit="m"` ↔ 训练 `depth_output_unit=m` ↔ env 米），无需补丁

## 本项目包含的内容

| 部分 | 路径 | 作用 |
|------|------|------|
| 仿真核心 | `src/mujoco_lerobot` | MuJoCo 封装、IK、相机渲染、数据采集全套 |
| 采集脚本 | `scripts/collect_data.py` | 一键数据采集（teacher / keyboard）入口 |
| 评估环境插件 | `src/lerobot_env_mujoco_lerobot` | gymnasium 环境 + EnvConfig，供 `lerobot-eval` 使用 |
| 策略插件 | `src/lerobot_policy_Adaptive_ACT` | 自适应 ACT（`adaptive_act`），供 `lerobot-train` 使用 |
| 渲染库 | `libs/mujoco_camrender` | C++ 多相机并行离屏渲染引擎（submodule） |
| 配置文件 | `configs/` | 任务 / 场景 / 数据集 / teacher / 策略 YAML |

### 仿真核心 `src/mujoco_lerobot`

主包（PyPI 名 `mujoco-lerobot-core`），包含：

- `configs/`：YAML 配置加载
  - `config_loader.py`：任务 / 场景配置加载、`resolve_config_path`
  - `dataset_config.py`：`DatasetConfig`，解析 `use_recode_scale` 多速率 / 扁平记录格式
  - `teacher_config.py`、`paths.py`
- `simulate/`：底层仿真
  - `mujoco_wrapper.py`：MuJoCo 模型 / 数据管理
  - `ik_solver.py`：基于 mink 的 IK 解算
  - `camera_renderer.py`：相机渲染（camrender 并行 / 回退 `mujoco.Renderer`）
  - `actuators.py`：执行器控制
- `data/`：数据采集
  - `simulation_manager.py`：episode 调度（物理步进、帧回调、视图）
  - `controllers.py`：`ScriptedTeacherController` / `KeyboardTeleopController`
  - `teachers/`：scripted teacher 实现（`pick_place_teacher.py` / `dual_pick_place_teacher.py` / `base.py`）
  - `observation_collector.py`：观测采集（state / action / 相机帧）
  - `dataset_writer.py`：`LeRobotDatasetWriter`（流式写入、追加一致性校验）
  - `reset_manager.py`：复位与物体随机化

### 数据采集脚本 `scripts/collect_data.py`

一键采集入口：`--mode teacher`（自动，默认）/ `--mode keyboard`（键盘遥操作）。主要参数：

| 参数 | 说明 |
|------|------|
| `--task-config` / `--task` | 任务配置文件 / 任务名（`--task list` 列出所有） |
| `--dataset-config` | 数据集记录格式配置 |
| `--mode` | `teacher`（自动）或 `keyboard`（键盘） |
| `--episodes` | 目标 episode 数（`--append` 时为**本轮新增**条数） |
| `--repo-id` | 数据集 repo_id，默认 `mujoco_<任务名>`（如 `mujoco_pick_place`） |
| `--output` | 输出根目录（默认 `outputs/datasets`），写入 `<output>/<任务名>/<时间戳>` |
| `--output-dir` | **直接指定**数据集输出目录（精确路径，不再附加任务名 / 时间戳子目录） |
| `--append DIR` | 在已有数据集上**追加** episode（追加前校验配置一致） |
| `--max-time` | 每 episode 仿真时间上限（秒），默认取 `simulate_default.yaml` |
| `--no-render` | 无头模式（不打开 viewer；无 DISPLAY 时自动 EGL 离屏渲染） |
| `--overwrite` | 覆盖已存在的输出目录 |

### LeRobot 评估环境插件 `src/lerobot_env_mujoco_lerobot`

gymnasium 环境 + `EnvConfig` 注册，供 `lerobot-eval --env.type=mujoco_lerobot` 使用：

- `lerobot_env.py`：`MujocoLerobotEnv`，任务无关（`--env.task` / `--env.dataset_config` 由 CLI 传入）
- `lerobot_env_cfg.py`：EnvConfig 注册 + 观测 preprocessor
  - rgb：uint8 `[0,255]` → `[0,1]`（对齐训练 dataloader 的 `/255`）
  - depth：按 `--env.depth_output_unit`（默认 `m`）把 env 米制深度转到训练使用的单位
- 环境产出 gym 风格键（`state.*` / `images.*`），任何在该数据集格式上训练的策略可直接评估；
  其他策略可覆盖 `get_env_processors` 提供外部观测数据处理器适配

### 自适应 ACT 策略插件 `src/lerobot_policy_Adaptive_ACT`

基于原生 ACT（Action Chunking Transformer）扩展的策略插件，注册为 LeRobot 策略类型
`adaptive_act`（`--policy.type=adaptive_act`），供 `lerobot-train` 直接训练。实现
分布在三个文件：`configuration_adaptive_act.py`（`AdaptiveACTConfig`，超参）、
`modeling_adaptive_act.py`（`AdaptiveACTPolicy`，模型结构）、`processor_adaptive_act.py`
（输入 / 输出处理器）。模型基于纯 `torch.nn.Transformer` 构建，任务无关（不限于
pick_place）。

**输入适应性** —— 解决"任意相机配置都能直接训练"的问题：

- **任意图像通道数**：支持灰度（1 通道）、RGB（3 通道）、RGB-D（4 通道）及任意
  通道数；非 3 通道时仅替换 resnet 的 `conv1`（灰度取预训练 RGB 均值、多余通道取
  RGB 均值），**其余层保留 ImageNet 预训练权重**。
- **通道数自动识别**：`image_channels=None`（默认）时从数据集 `input_features`
  自动识别通道数，无需人工指定。
- **RGBD 支持（通道拼接）**：`concat_visual_features` 把同一相机的 rgb（3ch）+
  depth（1ch）沿通道维拼成 4 通道 RGBD 视图（conv1 自动适配），无需重新采集数据集；
  拼接视图通道数 = 各源通道数之和，拼接前各源特征按自身 stats 归一化。

**多相机组织** —— 按相机分组共享 / 独立 backbone：

- `camera_backbone_groups` 指定各分组包含的相机：同组相机共用一个 resnet18，
  不同组用独立 resnet18；未指定的相机自动归入共享默认组。
  例如 `{hand: [camera_left, camera_right], global: [camera_top]}` —— 两个手眼
  相机共用一个 backbone，全局相机用另一个。

**训练配置** —— 全部超参可经 YAML 配置文件或命令行传入（命令行优先，覆盖同名参数）：

- 完整训练配置见 `configs/policy/adaptive_act.yaml`（含 `policy:` 超参 + `dataset:`
  + 训练参数），训练命令见「策略训练」章节。
- RGBD 用法示例（已内置在配置中）：`input_features` 同时保留 rgb 与 depth，并通过
  `concat_visual_features` 把它们拼成一个视图：

```yaml
policy:
  type: adaptive_act
  input_features:
    observation.state: {type: STATE, shape: [7]}
    observation.images.realsense_link_CAMERA.rgb:   {type: VISUAL, shape: [3, 480, 640]}
    observation.images.realsense_link_CAMERA.depth: {type: VISUAL, shape: [1, 480, 640]}
  concat_visual_features:
    observation.images.realsense_link_CAMERA.rgbd:
      - observation.images.realsense_link_CAMERA.rgb
      - observation.images.realsense_link_CAMERA.depth
  # 拼接后有效视觉输入为 4 通道 RGBD，conv1 自动适配，image_channels 无需指定
```

### 多相机渲染库 `libs/mujoco_camrender`

C++ + pybind11 的 MuJoCo 多相机并行离屏渲染引擎（git submodule）：

- 每相机独立线程并行渲染，RGB + 深度一次完成；深度通过透视投影公式从深度缓冲线性化恢复
- GLFW（有 DISPLAY）/ EGL（无头）双后端，CMake 自动检测
- 详细文档见 [libs/mujoco_camrender/README.md](libs/mujoco_camrender/README.md)

### 配置文件 `configs`

- `tasks/tasks.yaml`：任务配置 —— 通过 `scene_file` / `scene_config_file` / `teacher_config_file`
  连接场景与 teacher；**不**引用数据集配置（同一场景可有多种采集格式）
- `scenes/scene_*.yaml`：机器人（前缀、关节、末端 site）与相机（名称、fps、分辨率），
  以及评估视角 `view:`（`lookat` / `distance` / `elevation` / `azimuth` / `image_size`）
- `dataset/dataset_*.yaml`：数据记录格式（详见「数据采集 → 配置说明」）
- `teachers/*.yaml`：scripted teacher 参数（对象名、阈值、高度、夹爪指令等）
- `policy/adaptive_act.yaml`：自适应 ACT 完整训练配置（`policy:` 超参 + `dataset:` + 训练参数）
- `simulate_default.yaml`：仿真 / 控制频率（`sim.physics_dt` / `sim.policy_dt`）与采集参数（`collection.*`）

## 环境准备

```bash
# 0. 初始化子模块（mujoco_camrender 为 git submodule）
git submodule update --init --recursive

# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装依赖
uv sync

# 3. 构建 mujoco-camrender（C++ 库 + Python 绑定）
#    需要系统 MuJoCo（>=3.11.0）、CMake、GLFW、OpenGL 头文件
cd libs/mujoco_camrender
mkdir -p build && cd build
cmake .. -DMUJOCO_DIR=$HOME/Softwares/mujoco-3.11.0
make -j$(nproc)
cp src/pybind/_mujoco_camrender.cpython-312-x86_64-linux-gnu.so ../src/mujoco_camrender/
cd ../../..
uv sync --reinstall-package mujoco-camrender

# 4. 安装 ffmpeg :
sudo apt install ffmpeg -y
```

> `.so` 已内嵌 RUNPATH 指向 camrender 构建目录与 MuJoCo 库目录，通常无需手动设置
> `LD_LIBRARY_PATH`；如需覆盖见 `.env.example`。

## 数据采集

### 自动采集（scripted teacher）

单臂 pick_place：

```bash
uv run python scripts/collect_data.py \
    --task-config configs/tasks/tasks.yaml \
    --task pick_place \
    --dataset-config configs/dataset/dataset_pick_place.yaml \
    --episodes 100 \
    --no-render
```

双臂 pick_place：

```bash
uv run python scripts/collect_data.py \
    --task-config configs/tasks/tasks.yaml \
    --task dual_pick_place \
    --dataset-config configs/dataset/dataset_dual_pick_place.yaml \
    --episodes 100 \
    --no-render
```

### 键盘遥操作采集（打开 MuJoCo viewer，需要真实桌面显示环境）

```bash
uv run python scripts/collect_data.py \
    --task-config configs/tasks/tasks.yaml \
    --task pick_place \
    --dataset-config configs/dataset/dataset_pick_place.yaml \
    --mode keyboard
```

键盘控制：`W/S A/D R/F` = x/y/z 平移，方向键 / `Q/E` = 旋转，`Space` = 夹爪开合，
`Enter` = 保存当前 episode，`Backspace` = 丢弃，`Esc` = 退出。

### 自定义 repo_id 与数据集目录

```bash
# 指定 repo_id 与输出目录（--output-dir 为精确目录，不再附加 <任务名>/<时间戳>）
uv run python scripts/collect_data.py \
    --task-config configs/tasks/tasks.yaml \
    --task pick_place \
    --dataset-config configs/dataset/dataset_pick_place.yaml \
    --repo-id my_org/my_pick_place \
    --output-dir outputs/datasets/my_pick_place \
    --episodes 100 \
    --no-render
```

不传 `--output-dir` 时，数据集写入 `<--output>/<任务名>/<时间戳>`（如
`outputs/datasets/pick_place/20260809_154748`）。

### 追加数据（`--append`）

```bash
uv run python scripts/collect_data.py \
    --task-config configs/tasks/tasks.yaml \
    --task pick_place \
    --dataset-config configs/dataset/dataset_pick_place.yaml \
    --append outputs/datasets/pick_place/20260809_154748 \
    --episodes 100 \
    --no-render
```

**`--append` 语义**：在已有数据集上**追加 `--episodes` 指定条数的新 episode**
（总量 = 已有 + 新增），**不是**"补到总量达到 `--episodes`"。追加前会校验当前
配置与已有数据集的 fps / features / 深度单位 / 编码参数一致，不一致会抛错拒绝追加。

### 无头（无 DISPLAY）环境

`--no-render` 在 SSH 服务器/无显示器环境下会自动使用 camrender 的 **EGL 后端**做
并行离屏渲染（相机 RGB/深度照常采集），无需 X 服务。要求 GPU 驱动支持 EGL（NVIDIA
驱动自带）；若不可用，自动回退 `mujoco.Renderer` + EGL。有显示器时使用默认 GLFW
（viewer 正常显示）。

### 配置说明

- `configs/tasks/tasks.yaml`：任务配置，通过 `scene_config_file` / `teacher_config_file`
  连接场景与 teacher 配置；**不**引用数据集配置（同一场景可有多种采集格式）。
- `configs/scenes/scene_*.yaml`：机器人（前缀、关节、末端 site）与相机
  （名称、fps、分辨率）。相机参数与 mjcf 冲突时会打印警告。
  **深度范围不在此定义**（见下方数据集配置）。
- `configs/dataset/dataset_*.yaml`：数据记录格式。全局开关 `use_recode_scale`：
  - `true`：启用多速率采样，非相机数据按 `recode_hz × recode_scale` 扩展为一帧内的
    多个子采样（shape `(R, D)`），适用于自定义多速率模型（如 BiMFT）
  - `false`：所有非相机数据按 `recode_hz` 记录，忽略所有 `recode_scale`，产出
    **LeRobot 标准扁平格式**：state 拼接为 `observation.state (state_dim,)`、
    action 为 `(action_dim,)`，**可直接用于 LeRobot 内置 ACT** 策略
  - 相机数据固定按 `recode_hz` 记录；渲染参数（`image_size`/`fps`）在场景配置中
  - `state.camera.depth_range`：深度范围（米），**仅** lerobot 深度图像编码器使用
  - 其余可注释启用的数据源：joint `velocity`/`effort`、sensor `gyro`/
    `accelerometer`、`frame`/`action.tcp` 的 `position`/`euler`/`quat`
    （支持 `frame_site` 参考系与 `type`：`absolute`/`relative`/`velocity`）
- `configs/teachers/*.yaml`：scripted teacher 参数（对象名、阈值、高度、夹爪指令等）。

## 策略训练（Adaptive ACT）

使用内置自适应 ACT 策略插件（`--policy.type=adaptive_act`）在采集的数据集上训练
（`use_recode_scale: false` 的扁平格式可直接训练）。策略参数可经
`configs/policy/adaptive_act.yaml` 配置文件提供，也可全部用命令行传入；命令行
参数优先级更高，会覆盖配置文件中的同名参数。

**输入特征注意**：数据集同时含 rgb（3 通道）与 depth（1 通道）两个 VISUAL 特征。
可选两种用法：
- 纯 RGB：`input_features` 只选 rgb（3 通道）；
- RGBD：把同一相机的 rgb+depth 通过 `concat_visual_features` 沿通道维拼成
  4 通道输入（见 `configs/policy/adaptive_act.yaml`）。
图像通道数会自动从特征 shape 识别（`image_channels` 无需指定）。

**深度单位注意**：本项目深度单位统一为**米**（MuJoCo 米制渲染）。记录侧采集时在
features info 里显式写入 `depth_unit="m"`（`info.json` 记录单位）；训练侧
`configs/policy/adaptive_act.yaml` 显式设 `dataset.depth_output_unit: m` 与之对齐
（评估 env 亦产出米）。两端显式统一，无需任何补丁/自动识别。若确需强制覆盖，显式传
`--dataset.depth_output_unit=<m|mm>`（此时评估 env 需同步设 `--env.depth_output_unit`）即可。

```bash
# 使用策略配置文件 config_path（推荐；命令行可覆盖任意字段，如 dataset.root / steps）
uv run lerobot-train \
    --num_workers=12 \
    --prefetch_factor=4 \
    --persistent_workers=true \
    --save_freq=10000 \
    --steps=50000 \
    --batch_size=48 \
    --config_path=configs/policy/adaptive_act.yaml \
    --dataset.root=$(pwd)/outputs/datasets/<数据集名>/<时间戳目录>
```

### 多 GPU 训练

`lerobot-train` 内部使用 `accelerate.Accelerator`（自动检测分布式），因此多 GPU 只需
**用 accelerate 启动器运行**：

> **⚠️ 多 GPU 必须用以下两个关键参数**（缺一不可，见下方「根因」）：
```bash
export NCCL_IB_DISABLE=1
```
> 并在命令加 `--dataloader_multiprocessing_context=fork`：
```bash
env NCCL_IB_DISABLE=1 \
uv run accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    --config_path=configs/policy/adaptive_act.yaml \
    --num_workers=8 \
    --prefetch_factor=8 \
    --persistent_workers=true \
    --save_freq=10000 \
    --steps=50000 \
    --batch_size=48 \
    --dataset.root=$(pwd)/outputs/datasets/<数据集名>/<时间戳目录>
```

**注意**：

- `--batch_size` 是**每 GPU** 的：accelerate 把每个 batch 分片到各 rank，有效全局
  batch = `batch_size × num_processes`。2 卡下 `--batch_size=32` → 全局 64；
  若想保持全局 32，用 `--batch_size=16`。
- 混合精度：`AdaptiveACTConfig`（同原生 ACT）**没有 `dtype` 字段**，不能传
  `--policy.dtype=...`。直接用启动器参数 `--mixed_precision=bf16`；lerobot 会对无 `dtype` 字段的策略把 `mixed_precision` 设为 `None`，
  accelerate 会回退读取 `ACCELERATE_MIXED_PRECISION` 环境变量，因此 `bf16` 生效
  （已验证两 rank 均为 bf16）。想要全精度则用 `--mixed_precision=no`。
- 分布式下 `--policy.device` 会被忽略（accelerate 自动按 rank 分配 GPU）。
- env 评估只在主进程执行；checkpoint 会记录 `num_processes`/`batch_size`，续训自动恢复。

### 继续训练（resume）

中断或想延长训练时，从已有 checkpoint 续训。checkpoint 目录结构为
`<output_dir>/checkpoints/<step>/`（含 `pretrained_model/` 与 `training_state/`），
`checkpoints/last/` 为最新一步的软链接。

```bash
# 单 GPU 续训：从 checkpoints/last 恢复模型权重 + 优化器 + 调度器 + RNG 状态
uv run lerobot-train \
    --config_path=outputs/train/<时间戳目录>/<时间戳+策略名>/checkpoints/last/pretrained_model \
    --resume=true \
    --output_dir=outputs/train/<时间戳目录>/<时间戳+策略名> \
    --steps=<新总步数>
```

**参数说明**：

- `--config_path`：指向 checkpoint 的 `pretrained_model/` 目录（或其中的
  `train_config.json`）。续训时**默认使用 checkpoint 里保存的完整训练配置**
  （`train_config.json`，含 policy 超参 / dataset / 优化器 / 调度器等），命令行
  `--*` 参数仍可覆盖同名项。
- `--resume=true`：开启续训模式（必须与 `--config_path` 搭配）。
- `--output_dir`：**必须指向原训练输出目录**（与首次训练一致）。若省略，lerobot
  会新建一个 `<时间戳>_resume` 目录，导致续训结果与原始 run 分离。
- `--steps`：**新的总步数，不是增量**。训练从已恢复的步数继续跑到 `steps`：
  例如原训练 100000 步，想再训 50000 步，设 `--steps=150000`。
- 数据相关参数（`--dataset.root` 等）默认从 checkpoint 配置恢复；如需换数据集
  继续训练，显式覆盖 `--dataset.root=<新数据集目录>` 即可（注意特征 / 单位须一致）。

**多 GPU 续训**：与首次训练相同，需 `NCCL_IB_DISABLE=1` +
`--dataloader_multiprocessing_context=fork`，且 `--num_processes` / `--batch_size`
应与原训练一致——checkpoint 记录了这两项，续训时用于**样本级精确衔接**
（数据顺序从断点处继续，不重复不跳过）：

```bash
env NCCL_IB_DISABLE=1 \
uv run accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    --config_path=outputs/train/<时间戳目录>/<时间戳+策略名>/checkpoints/last/pretrained_model \
    --resume=true \
    --output_dir=outputs/train/<时间戳目录>/<时间戳+策略名> \
    --steps=<新总步数>
```


## 策略评估（lerobot-eval）

环境观测键直接使用数据集格式（`observation.state.*` / `observation.images.*`），
任何在该格式上训练的策略可直接评估；其他策略可配合外部观测数据处理器
（覆盖 `get_env_processors`）适配。

```bash
# 无头评估（策略类型从 checkpoint 的 config.json 自动识别为 adaptive_act）
uv run lerobot-eval \
    --env.type=mujoco_lerobot \
    --env.task=pick_place \
    --env.dataset_config=configs/dataset/dataset_pick_place.yaml \
    --eval.n_episodes=10 \
    --eval.batch_size=10  \
    --eval.recording=true \
    --policy.path=outputs/train/<时间戳目录>/<时间戳+策略名>/checkpoints/last/pretrained_model
```


### 可视化评估（打开 MuJoCo viewer，需要真实桌面显示环境）

```bash
uv run lerobot-eval \
    --env.type=mujoco_lerobot \
    --env.task=pick_place \
    --env.dataset_config=configs/dataset/dataset_pick_place.yaml \
    --env.use_viewer=true \
    --eval.n_episodes=10 \
    --eval.batch_size=10  \
    --eval.recording=true \
    --policy.path=outputs/train/<时间戳目录>/<时间戳+策略名>/checkpoints/last/pretrained_model
```

### 评估参数说明

- `--eval.n_episodes`：评估（并统计成功率）的 episode 总数。
- `--env.max_episode_steps`：每个 episode 的最大仿真步数上限。
  每个 step 推进 `1 / recode_hz` 秒（默认 100Hz → 0.01s），本任务 teacher 演示
  平均约 9s（906 帧），因此评估时建议给足时间，例如
  `--env.max_episode_steps=4000`（约 40s），否则策略即使学会也来不及完成。
- **视频数量上限**：lerobot-eval 默认最多为**前 10 个** episode 渲染视频
  （`eval_episode_0.mp4` ~ `eval_episode_9.mp4`，位于
  `output_dir/videos/mujoco_lerobot_0/`）。**所有** `n_episodes` 都参与成功率
  统计，只是超过 10 个的 episode 不再生成视频。如需为全部 episode 生成视频，
  可开启 `--eval.recording`（此时不限视频数，但同时会把评估结果录制为
  LeRobot 数据集到 `output_dir/recordings/`，磁盘占用较大）。
- **视频帧率/视角**：eval 视频用场景配置 `view:` 的自由视角渲染
  （`lookat`/`distance`/`elevation`/`azimuth`），分辨率由 `view.image_size`
  控制；播放帧率 = `recode_hz`（100），因此视频时长 = 仿真时长（无慢放）。
- **评估侧输入归一化（与训练一致）**：评估时 env 产出的观测会经环境 preprocessor
  适配到与训练相同的表示后，再进策略 preprocessor（MEAN_STD，stats 来自数据集）：
  - rgb：uint8 `[0,255]` → `[0,1]`（对齐训练 dataloader 的 `/255`）；
  - depth：env 产出**米**，训练配置显式 `dataset.depth_output_unit=m`（记录侧
    `info.json` 亦显式记录 `depth_unit="m"`）解码、stats 亦为米 → 评估无需转换。
    若训练被设为 `"mm"`，需把 env preprocessor 设为 `--env.depth_output_unit=mm`
    （米×1000 转毫米）以对齐。
  若 rgb/depth 输入表示与训练分布不一致，策略在评估时"失明"、输出退化动作。
  两端单位显式统一（记录 `depth_unit="m"` ↔ 训练 `depth_output_unit=m` ↔ env 米），
  无补丁/自动识别。
- **物体随机化与 seed**：`MujocoLerobotEnv.reset(seed=...)` 会真正用该 seed 控制
  物体随机化（gym 语义）：同 seed 两次 reset 结果相同，不同 seed 不同。评估时
  `lerobot-eval` 为每个 episode 用 `seed + episode_index` 播种，因此各 episode
  随机化不同、且同一 episode 跨次评估可复现；若想换一组随机化，改 `--eval.seed`
  （或 `cfg.seed`）即可。

## 项目结构

```
configs/                    # YAML 配置
  ├── tasks/tasks.yaml      #   任务（scene / teacher 连接、物体随机化范围）
  ├── scenes/               #   机器人 / 相机 / 评估视角
  ├── dataset/              #   数据记录格式（use_recode_scale / features / depth_range）
  ├── teachers/             #   scripted teacher 参数
  ├── policy/               #   策略训练配置（adaptive_act.yaml）
  └── simulate_default.yaml #   仿真 / 控制频率、采集参数
assets/mujoco/              # MuJoCo 场景与机器人模型（scenes / robot / objects）
libs/mujoco_camrender/      # 多相机并行渲染库（C++ + pybind11，submodule）
scripts/collect_data.py     # 一键数据采集（teacher / keyboard）
src/mujoco_lerobot/         # 主包（仿真核心）
  ├── configs/              # 配置加载（含 use_recode_scale）
  ├── simulate/             # MuJoCo 封装、IK（mink）、相机渲染
  └── data/                 # 采集（采集器/控制器/teacher/写入器/调度器）
src/lerobot_env_mujoco_lerobot/          # LeRobot 评估环境插件（gym 环境 + EnvConfig）
src/lerobot_policy_Adaptive_ACT/         # 自适应 ACT 策略插件（adaptive_act）
tests/                      # 单元测试（env / 策略 / 诊断 rollout）
outputs/
  ├── datasets/<任务名>/<时间戳>/   # 采集的数据集
  ├── train/<时间戳>/               # 训练输出（checkpoints）
  └── eval/<日期>/                  # 评估输出（视频 / 录制）
```
