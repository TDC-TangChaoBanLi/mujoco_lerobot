"""诊断 rollout：完整复现 lerobot-eval 数据流，观察策略动作是否合理。

数据流（与 lerobot_eval.rollout 一致）:
    preprocess_observation → env_preprocessor → policy preprocessor
    → policy.select_action → policy postprocessor → env_postprocessor

用于排查「成功率 0」是训练不足还是观测/动作链路问题。
"""

from __future__ import annotations

import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.envs.utils import preprocess_observation
from mujoco_lerobot.env.lerobot_env import MujocoLerobotEnv
from mujoco_lerobot.env.lerobot_env_cfg import MujocoLerobotEnvConfig

CKPT = "outputs/train/act_pick_place/checkpoints/last/pretrained_model"


def main() -> None:
    env = MujocoLerobotEnv(
        task_name="pick_place",
        dataset_config="configs/dataset/dataset_pick_place.yaml",
    )

    from lerobot.configs.policies import PreTrainedConfig

    policy_cfg = PreTrainedConfig.from_pretrained(CKPT)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg, pretrained_path=CKPT
    )
    env_preprocessor, env_postprocessor = MujocoLerobotEnvConfig(
        task="pick_place",
        dataset_config="configs/dataset/dataset_pick_place.yaml",
    ).get_env_processors()
    policy = ACTPolicy.from_pretrained(CKPT)
    policy.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)

    obs, _ = env.reset(seed=42)
    policy.reset()  # 清空动作队列
    cube_id = env._success_obj["cube"]
    plate_id = env._success_obj["plate"]

    def obj_xy():
        return (
            env._mj.data.xpos[cube_id].copy(),
            env._mj.data.xpos[plate_id].copy(),
        )

    c0, p0 = obj_xy()
    print(f"init cube={c0.round(3)} plate={p0.round(3)}")

    ee_site = env._scene_cfg.robots[0].prefixed_ee_site
    dists = []
    for step in range(80):
        observation = preprocess_observation(obs)
        observation["task"] = ["pick and place a red cube onto a blue plate"]
        observation = env_preprocessor(observation)

        observation = preprocessor(observation)

        with torch.inference_mode():
            action = policy.select_action(observation)

        action = postprocessor(action)
        action = action[ACTION] if isinstance(action, dict) else action
        action_np = np.asarray(action.detach().cpu()).reshape(-1)

        obs, reward, terminated, truncated, info = env.step(action_np)
        c, p = obj_xy()
        ee = env._mj.get_site_pose(ee_site)
        dists.append(np.linalg.norm(c[:2] - p[:2]))
        if step % 5 == 0 or step == 79:
            print(
                f"step={step:3d} act={np.round(action_np, 3)} "
                f"ee={np.round(ee[:3], 3)} cube={np.round(c, 3)} "
                f"dist={np.linalg.norm(c[:2]-p[:2]):.3f}"
            )
        if terminated or truncated:
            print(f"done at step {step}: terminated={terminated} truncated={truncated}")
            break

    print(f"min xy dist reached: {min(dists):.3f}")
    env.close()


if __name__ == "__main__":
    main()
