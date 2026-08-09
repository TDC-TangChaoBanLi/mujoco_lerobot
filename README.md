# mujoco-lerobot

基于 MuJoCo 的 LeRobot 仿真环境：数据采集 + 策略评估。

- 自动（scripted teacher）/ 键盘遥操作数据采集，由 YAML 配置驱动
- 高性能多相机并行渲染（[mujoco_camrender](libs/mujoco_camrender)，RGB + 深度线性化）
- IK 解算基于 [mink](https://github.com/kevinzakka/mink/)
- LeRobot 环境插件（`lerobot_env_mujoco_lerobot`），支持 `lerobot-eval` 直接评估
- 自适应 ACT 策略插件（`lerobot_policy_Adaptive_ACT`）：任意图像通道数输入 +
  按相机分组共享 / 独立 resnet18 backbone，注册为策略类型 `adaptive_act`

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

```bash
# 自动采集（scripted teacher）
uv run python scripts/collect_data.py \
    --task-config configs/tasks/tasks.yaml \
    --task pick_place \
    --dataset-config configs/dataset/dataset_pick_place.yaml \
    --episodes 50 \
    --no-render

# 双臂自动采集
uv run python scripts/collect_data.py \
    --task-config configs/tasks/tasks.yaml \
    --task dual_pick_place \
    --dataset-config configs/dataset/dataset_dual_pick_place.yaml \
    --episodes 50\
    --no-render

# 键盘遥操作采集（打开 MuJoCo viewer）
uv run python scripts/collect_data.py \
    --task-config configs/tasks/tasks.yaml \
    --task pick_place \
    --dataset-config configs/dataset/dataset_pick_place.yaml \
    --mode keyboard
```

> **无头（无 DISPLAY）环境**：`--no-render` 在 SSH 服务器/无显示器环境下会自动使用
> **camrender 的 EGL 后端**做并行离屏渲染（相机 RGB/深度照常采集），无需 X 服务。
> 要求 GPU 驱动支持 EGL（NVIDIA 驱动自带）；若不可用，自动回退 `mujoco.Renderer` + EGL。
> 有显示器时使用默认 GLFW（viewer 正常显示）。

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

## 策略训练（Adaptive ACT + 离线 WandB）

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
# 方式一：使用策略配置文件（推荐；命令行可覆盖任意字段，如 dataset.root / steps）
uv run lerobot-train \
    --config_path=configs/policy/adaptive_act.yaml \
    --dataset.repo_id=mujoco_pick_place \
    --dataset.root=$(pwd)/outputs/datasets/<数据集名>/<时间戳目录> \
    --steps=100000 \
    --batch_size=32 \
    --num_workers=8

# 方式二：全部命令行参数
uv run lerobot-train \
    --policy.type=adaptive_act \
    --policy.push_to_hub=false \
    --policy.input_features='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.realsense_link_CAMERA.rgb":{"type":"VISUAL","shape":[3,480,640]}}' \
    --dataset.repo_id=mujoco_pick_place \
    --dataset.root=$(pwd)/outputs/datasets/<数据集名>/<时间戳目录> \
    --output_dir=outputs/train/adaptive_act_pick_place \
    --steps=100000 \
    --batch_size=32 \
    --num_workers=8 \
    --save_freq=5000 \
    --log_freq=100
```

### 多 GPU 训练

`lerobot-train` 内部使用 `accelerate.Accelerator`（自动检测分布式），因此多 GPU 只需
**用 accelerate 启动器运行**，无需改任何代码。本机示例（2× RTX 3090）：

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
    --dataset.repo_id=mujoco_pick_place \
    --dataset.root=$(pwd)/outputs/datasets/<数据集名>/<时间戳目录> \
    --steps=100000 \
    --batch_size=32 \
    --num_workers=8 \
    --dataloader_multiprocessing_context=fork
```

等价的 `torchrun` 方式：

```bash
env NCCL_IB_DISABLE=1 \
uv run torchrun \
    --nproc_per_node=2 \
    -m lerobot.scripts.lerobot_train \
    --config_path=configs/policy/adaptive_act.yaml \
    --dataset.repo_id=mujoco_pick_place \
    --dataset.root=$(pwd)/outputs/datasets/<数据集名>/<时间戳目录> \
    --steps=100000 \
    --batch_size=32 \
    --num_workers=4 \
    --dataloader_multiprocessing_context=fork
```

**注意**：

- `--batch_size` 是**每 GPU** 的：accelerate 把每个 batch 分片到各 rank，有效全局
  batch = `batch_size × num_processes`。2 卡下 `--batch_size=32` → 全局 64；
  若想保持全局 32，用 `--batch_size=16`。
- 混合精度：`AdaptiveACTConfig`（同原生 ACT）**没有 `dtype` 字段**，不能传
  `--policy.dtype=...`。直接用启动器参数 `--mixed_precision=bf16`（RTX 3090 Ampere
  支持）；lerobot 会对无 `dtype` 字段的策略把 `mixed_precision` 设为 `None`，
  accelerate 会回退读取 `ACCELERATE_MIXED_PRECISION` 环境变量，因此 `bf16` 生效
  （已验证两 rank 均为 bf16）。想要全精度则用 `--mixed_precision=no`。
- 分布式下 `--policy.device` 会被忽略（accelerate 自动按 rank 分配 GPU）。
- env 评估只在主进程执行；checkpoint 会记录 `num_processes`/`batch_size`，续训自动恢复。


### 启用离线 WandB（本地记录指标曲线）

lerobot 训练默认只把指标打印到终端（不落盘、不支持 TensorBoard）。要持久化
loss/学习率等曲线，可启用 WandB **离线模式**——不需要登录账号，日志写入
`output_dir/wandb/` 目录，之后可手动同步到云端查看：

```bash
uv run lerobot-train \
    --config_path=configs/policy/adaptive_act.yaml \
    --dataset.root=$(pwd)/outputs/datasets/pick_place/<时间戳目录> \
    --output_dir=outputs/train/adaptive_act_pick_place \
    --steps=100000 \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=mujoco_lerobot
```

#### WandB 参数说明

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `--wandb.enable` | `true` | 是否启用 WandB 日志（默认 `false`，仅打印到终端） |
| `--wandb.mode` | `offline` | 日志模式：`offline`（本地，无需登录）/ `online`（需 `wandb login`）/ `disabled`（关闭） |
| `--wandb.project` | `mujoco_lerobot` | 项目名，用于云端分组 |
| `--wandb.entity` | 可选 | 云端组织/用户名（离线模式可省略） |
| `--wandb.run_id` | 可选 | 指定 run id（续训时自动复用） |

> **提示**：训练脚本不会自动重定向终端日志到文件。建议用 `nohup ... > train.log 2>&1 &`
> 保存终端指标，与 WandB 双保险。

#### 查看离线日志

离线运行的数据落在 `<output_dir>/wandb/`（如 `outputs/train/adaptive_act_pick_place/wandb/`），
每个 run 一个 `offline-run-<时间戳>-<run_id>` 目录，核心指标保存在目录内的
`run-<run_id>.wandb` 文件（LevelDB protobuf 格式）：

```bash
# 1) 列出离线 run
ls outputs/train/adaptive_act_pick_place/wandb/
#    → offline-run-20260806_210548-t0nynggt/
```

**方式 A：上传云端查看（推荐，可看曲线/图表）**

需先登录一次账号（无账号可先 `wandb signup`），然后同步上传，到 wandb.ai 网页查看：

```bash
wandb login            # 首次登录（一次性）
wandb sync outputs/train/adaptive_act_pick_place/wandb/offline-run-*   # 上传该 run 到云端
# 上传完成后控制台会打印云端 URL，打开即可查看 loss / lr / 梯度等曲线
```

> `wandb offline` 是「让后续 `python train.py` 以离线模式记录」的环境切换命令，
> 不是查看工具；离线日志一律用 `wandb sync` 上传后在网页查看。

**方式 B：本地解析 `.wandb` 文件（无需登录、无需上传）**

```bash
python - <<'EOF'
import glob
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2

for p in sorted(glob.glob("outputs/train/adaptive_act_pick_place/wandb/offline-run-*/run-*.wandb")):
    print(f"== {p} ==")
    ds = DataStore()
    ds.open_for_scan(p)
    rec = wandb_internal_pb2.Record()
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") == "request":
            req = rec.request
            if req.WhichOneof("request_type") == "log":
                d = {k: v for k, v in req.log.data.items() if not k.startswith("_")}
                if d:
                    print("step", req.log.step, d)
        rec.Clear()
    ds.close()
EOF
```

> 训练中途或完成后都可以运行上面的脚本；`wandb sync` 与本地解析脚本可同时使用。

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
    --policy.path=outputs/train/adaptive_act_pick_place/checkpoints/last/pretrained_model \
    --eval.n_episodes=5 \
    --eval.batch_size=1
```


# 可视化评估（打开 MuJoCo viewer，需要真实桌面显示环境）

```bash
uv run lerobot-eval \
    --env.type=mujoco_lerobot \
    --env.task=pick_place \
    --env.dataset_config=configs/dataset/dataset_pick_place.yaml \
    --env.use_viewer=true \
    --policy.path=outputs/train/adaptive_act_pick_place/checkpoints/last/pretrained_model
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

### 自适应 ACT 策略（`adaptive_act`）

内置自适应 ACT 策略插件 `lerobot_policy_Adaptive_ACT`，在原生 ACT 基础上扩展：

1. **任意图像通道数输入**：支持灰度（1 通道）、RGB（3 通道）、RGB-D（4 通道）及
   任意通道数；非 3 通道时仅替换 resnet 的 `conv1`（灰度取预训练 RGB 均值、多余
   通道取 RGB 均值），**其余层保留 ImageNet 预训练权重**。
2. **按相机分组共享 / 独立 backbone**：`camera_backbone_groups` 指定各分组包含的
   相机，同组相机共用一个 resnet18，不同组用独立 resnet18；未指定的相机自动归入
   共享默认组。
3. **通道数自动识别**：`image_channels=None`（默认）时从数据集的
   `input_features` 自动识别通道数，无需人工指定。
4. **RGBD 支持（通道拼接）**：`concat_visual_features` 可把同一相机的
   rgb（3ch）+ depth（1ch）沿通道维拼成 4 通道 RGBD 视图（conv1 自动适配），
   无需重新采集数据集；拼接视图的通道数 = 各源通道数之和。
5. **配置灵活**：全部超参可经 YAML 模型配置文件或命令行传入，命令行参数优先
   （覆盖配置文件中的同名参数）。
6. **全归一化**：所有输入 / 输出经 LeRobot 默认 MEAN_STD pre/post processor 归一化
   （拼接前各源特征按其自身 stats 归一化）。
7. 模型基于纯 `torch.nn.Transformer` 构建，任务无关（不限于 pick_place）。

```bash
# 训练（从数据集初始化，通道数/特征自动从 input_features 识别）
uv run lerobot-train \
    --policy.type=adaptive_act \
    --policy.input_features='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.realsense_link_CAMERA.rgb":{"type":"VISUAL","shape":[3,480,640]}}' \
    --dataset.repo_id=mujoco_pick_place \
    --dataset.root=outputs/datasets/pick_place/20260806_004855 \
    --output_dir=outputs/train/adaptive_act_pick_place \
    --steps=50000 --batch_size=32

# 评估
uv run lerobot-eval \
    --env.type=mujoco_lerobot --env.task=pick_place \
    --env.dataset_config=configs/dataset/dataset_pick_place.yaml \
    --policy.path=outputs/train/adaptive_act_pick_place/checkpoints/last/pretrained_model \
    --eval.n_episodes=5 --eval.batch_size=1 --eval.use_async_envs=false
```

模型配置使用完整训练配置文件 `configs/policy/adaptive_act.yaml`（含 `policy:`
自适应 ACT 超参、`dataset:` 与训练参数），命令行参数优先级更高：

```bash
# 用配置文件训练，命令行可覆盖任意字段
uv run lerobot-train \
    --config_path=configs/policy/adaptive_act.yaml \
    --policy.image_channels=4 \   # 命令行覆盖配置文件中的 image_channels
    --steps=100000
```

> 多相机分组示例见 `configs/policy/adaptive_act.yaml` 内注释：
> `camera_backbone_groups: {hand: [camera_left, camera_right], global: [camera_top]}`
> —— 两个手眼相机共用一个 resnet18，全局相机用另一个；未指定相机自动归入默认组。

RGBD（rgb+depth 4 通道）用法：在 `input_features` 中同时保留 rgb 与 depth，并通过
`concat_visual_features` 把它们拼成一个视图（`configs/policy/adaptive_act.yaml`
已内置该示例，可直接用于 pick_place）：

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

## 性能说明

- 物理步进 ~0.02ms、IK ~0.07ms，每帧开销极小；渲染是主要瓶颈。
- `mujoco_camrender` 并行渲染全部相机（RGB + 深度一次），深度通过透视投影
  公式从原始深度缓冲线性化恢复（`near=0.01·extent, far=50·extent`，已验证）。
- 数据写入使用 LeRobot 流式编码（后台线程），不缓存整集，内存恒定。
- **注意（WSL）**：在 WSL2 下 OpenGL 走 Mesa D3D12 翻译层，渲染较慢
  （单臂 1 相机约实时 30fps，双臂 3 相机约 11fps）；在原生 Linux GPU 上
  camrender 可达数百 FPS。MuJoCo viewer 需要真实桌面显示环境。

## 测试

```bash
# 单元测试（env / 策略 / 诊断 rollout）
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/
```

> 若系统安装了 ROS（`/opt/ros/*`），pytest 可能自动加载不兼容的 `launch_testing`
> 插件，用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 禁用第三方插件自动加载。

## 项目结构

```
configs/                    # YAML 配置（tasks / scenes / dataset / teachers）
assets/mujoco/              # MuJoCo 场景与机器人模型
libs/mujoco_camrender/      # 多相机并行渲染依赖
scripts/collect_data.py     # 一键数据采集（auto / keyboard）
src/mujoco_lerobot/         # 主包
  ├── configs/              # 配置加载（含 use_recode_scale）
  ├── simulate/             # MuJoCo 封装、IK、相机渲染
  └── data/                 # 采集（采集器/控制器/teacher/写入器/调度器）
src/lerobot_env_mujoco_lerobot/          # LeRobot 环境插件（gym 环境 + EnvConfig）
src/lerobot_policy_Adaptive_ACT/         # 自适应 ACT 策略插件
tests/                      # 单元测试（env / 策略 / 诊断 rollout）
```
