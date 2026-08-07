"""ExampleMLPPolicy — 消费 mujoco-lerobot 数据集格式观测的最小 MLP 策略。

仅用于演示 lerobot-eval 端到端流程；实际使用时请训练自己的策略。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION

from .configuration_example import ExampleMLPConfig


class ExampleMLPPolicy(PreTrainedPolicy):
    """最小 MLP 策略：展平状态 → 隐藏层 → 输出 flat 动作。"""

    config_class = ExampleMLPConfig
    name = "mujoco_lerobot_example"

    def __init__(
        self,
        config: ExampleMLPConfig,
        dataset_stats: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config)

        # 从 input_features 推断状态维度（取第一个 STATE 特征，展平 R*D）
        state_key = None
        state_dim = 0
        for key, feat in config.input_features.items():
            if feat.type.name == "STATE":
                state_key = key
                state_dim = int(feat.shape[0] * feat.shape[1])
                break
        if state_key is None:
            raise ValueError("ExampleMLPPolicy 需要至少一个 STATE 输入特征")
        self._state_key = state_key

        # 输出动作维度
        action_feat = config.output_features.get(ACTION)
        self._action_dim = int(action_feat.shape[0]) if action_feat else 7

        self.model = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, self._action_dim),
        )

    # ── LeRobot 必需接口 ─────────────────────────────

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(
        self, batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict | None]:
        state = batch[self._state_key].float().to(self._device())
        B = state.shape[0]
        state_flat = state.reshape(B, -1)
        action = self.model(state_flat)  # (B, action_dim)
        return action, None

    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        action, _ = self.forward(batch)  # (B, action_dim)
        # 输出为 (B, n_action_steps, action_dim)
        return action.unsqueeze(1).expand(
            action.shape[0], self.config.n_action_steps, action.shape[1]
        )

    def select_action(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        # 返回 (B, action_dim) —— 直接执行当前帧动作
        action, _ = self.forward(batch)
        return action

    def get_optim_params(self) -> dict:
        return {"params": self.model.parameters()}

    def reset(self) -> None:
        pass

    # ── 保存 / 加载 ───────────────────────────────────

    def _save_pretrained(
        self, save_directory: Path, state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        save_directory.mkdir(parents=True, exist_ok=True)
        sd = state_dict if state_dict is not None else self.model.state_dict()
        torch.save({"model": sd}, save_directory / "model.pt")

    @classmethod
    def _load_pretrained(
        cls, pretrained_path: Path, **kwargs,
    ) -> "ExampleMLPPolicy":
        from lerobot.utils.pretrained import load_pretrained

        return load_pretrained(cls, pretrained_path, **kwargs)
