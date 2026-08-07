"""诊断 rollout 测试 — 完整复现 lerobot-eval 数据流，观察策略动作是否合理。

原 ``scripts/debug_rollout.py`` 迁移而来，并修复了已失效的 ``env._success_obj``
引用（该属性在环境中不存在）。

数据流（与 lerobot_eval.rollout 一致）:
    preprocess_observation → env_preprocessor → policy preprocessor
    → policy.select_action → policy postprocessor → env_postprocessor → env.step

特点：
- 任务无关：task / dataset_config / checkpoint 均可通过环境变量覆盖，默认
  指向本仓库已有的 ACT pick_place 训练 checkpoint。
- 策略类型按 checkpoint 的 ``config.type`` 自动分发（原生 ``act`` 或本插件
  ``adaptive_act`` 均可）。
- 无 checkpoint 时自动跳过（``@pytest.mark.slow``）。

环境变量：
    MUJOCO_LEROBOT_CKPT     checkpoint 目录（默认 outputs/train/act_pick_place/
                            checkpoints/last/pretrained_model）
    MUJOCO_LEROBOT_TASK     任务名（默认 pick_place）
    MUJOCO_LEROBOT_DATASET_CONFIG  数据集配置（默认 configs/dataset/
                            dataset_pick_place.yaml）
    MUJOCO_LEROBOT_STEPS    rollout 步数上限（默认 80）
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION

from lerobot_env_mujoco_lerobot import (
    MujocoLerobotEnv,
    MujocoLerobotEnvConfig,
)

CKPT = os.environ.get(
    "MUJOCO_LEROBOT_CKPT",
    "outputs/train/act_pick_place/checkpoints/last/pretrained_model",
)
TASK = os.environ.get("MUJOCO_LEROBOT_TASK", "pick_place")
DATASET_CONFIG = os.environ.get(
    "MUJOCO_LEROBOT_DATASET_CONFIG",
    "configs/dataset/dataset_pick_place.yaml",
)
N_STEPS = int(os.environ.get("MUJOCO_LEROBOT_STEPS", "80"))


@pytest.mark.slow
@pytest.mark.skipif(
    not Path(CKPT).is_dir(), reason=f"checkpoint 不存在: {CKPT}（可用 "
    f"MUJOCO_LEROBOT_CKPT 指定）"
)
def test_diagnostic_rollout():
    """端到端 rollout：复现 eval 数据流并断言动作有限、环境可步进。"""
    env = MujocoLerobotEnv(task_name=TASK, dataset_config=DATASET_CONFIG)

    from lerobot.configs.policies import PreTrainedConfig

    policy_cfg = PreTrainedConfig.from_pretrained(CKPT)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg, pretrained_path=CKPT
    )
    env_preprocessor, env_postprocessor = MujocoLerobotEnvConfig(
        task=TASK, dataset_config=DATASET_CONFIG
    ).get_env_processors()

    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(CKPT)
    policy.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)

    obs, _ = env.reset(seed=42)
    policy.reset()  # 清空动作队列 / 时序集成状态

    action_norms: list[float] = []
    done = False
    step = 0
    terminated = truncated = False

    while step < N_STEPS and not done:
        observation = preprocess_observation(obs)
        observation["task"] = [env.task_description]
        observation = env_preprocessor(observation)
        observation = preprocessor(observation)

        with torch.inference_mode():
            action = policy.select_action(observation)

        action = postprocessor(action)
        action = action[ACTION] if isinstance(action, dict) else action
        action_np = np.asarray(action.detach().cpu()).reshape(-1)
        action_norms.append(float(np.linalg.norm(action_np)))

        obs, reward, terminated, truncated, info = env.step(action_np)
        step += 1

        # 动作必须有限（NaN/Inf 说明输入分布或归一化有误）
        assert np.isfinite(action_np).all(), f"step {step}: 动作含 NaN/Inf"

        if step % 10 == 0 or terminated or truncated:
            print(
                f"step={step:3d} act_norm={action_norms[-1]:.3f} "
                f"reward={reward:.1f} success={info.get('is_success', False)}"
            )
        done = bool(terminated or truncated)

    env.close()

    # 诊断结论
    print(
        f"rollout done at step {step}: terminated={terminated} "
        f"truncated={truncated} is_success={bool(env._check_success())}"
    )
    print(f"action norm min/mean/max = "
          f"{min(action_norms):.3f}/{float(np.mean(action_norms)):.3f}/{max(action_norms):.3f}")

    # 核心断言：至少执行过若干步且动作非退化（全 0 说明策略/链路失效）
    assert step >= 5, "rollout 过早结束，数据链路可能有问题"
    assert max(action_norms) > 1e-6, "动作恒为 0，策略未产出有效动作"
