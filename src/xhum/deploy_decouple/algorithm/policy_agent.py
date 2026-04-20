#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ACT policy inference (LeRobot v0.5.1) — runs under Python 3.12+ only.

Lives under ``src/xhum/deploy_decouple/algorithm`` so ROS (Python 3.10) never
imports LeRobot here. Keep in sync with ``src/xhum/deploy/policy_agent.py``
when the model I/O contract changes.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.policies.act.modeling_act import ACTPolicy


class PolicyAgent:
    """Thin wrapper around a pretrained ACTPolicy for real-robot inference."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.policy = self._load_policy()
        self.device = next(self.policy.parameters()).device

        self._image_keys: dict[str, tuple[int, int]] = {}
        self._state_key: str | None = None
        self._state_dim: int = 0
        self._action_dim: int = 0
        self._parse_model_config()

    def _load_policy(self) -> ACTPolicy:
        print(f"[PolicyAgent] Loading model from: {self.model_path}")
        policy = ACTPolicy.from_pretrained(self.model_path)
        device = policy.config.device
        print(f"[PolicyAgent] Model loaded on {device}")
        return policy

    def _parse_model_config(self):
        cfg = self.policy.config

        for key, feat in cfg.input_features.items():
            if feat.type is FeatureType.VISUAL:
                _, h, w = feat.shape
                self._image_keys[key] = (w, h)
            elif feat.type is FeatureType.STATE:
                self._state_key = key
                self._state_dim = feat.shape[0]

        for _key, feat in cfg.output_features.items():
            if feat.type is FeatureType.ACTION:
                self._action_dim = feat.shape[0]

        cam_list = ", ".join(f"{k} {v}" for k, v in self._image_keys.items())
        print(f"[PolicyAgent] Cameras  : {cam_list}")
        print(f"[PolicyAgent] State dim: {self._state_dim}  Action dim: {self._action_dim}")

    def inference(self, obs: dict) -> torch.Tensor:
        batch = self._prepare_batch(obs)
        return self.policy.select_action(batch)

    def reset(self):
        self.policy.reset()

    def _prepare_batch(self, obs: dict) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}

        for model_key, (target_w, target_h) in self._image_keys.items():
            cam_name = model_key.rsplit(".", maxsplit=1)[-1]
            img = obs["images"][cam_name]
            img = cv2.resize(img, dsize=(target_w, target_h))
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            batch[model_key] = img_t.unsqueeze(0).to(self.device, non_blocking=True)

        if self._state_key is not None:
            state = obs["arm_gripper_joints"]
            state_t = torch.from_numpy(np.asarray(state, dtype=np.float32))
            batch[self._state_key] = state_t.unsqueeze(0).to(self.device, non_blocking=True)

        return batch
