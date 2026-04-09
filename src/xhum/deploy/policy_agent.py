#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Policy inference wrapper for TienKung robot deployment.

Compatible with LeRobot v0.5.1 ACT policy API.  Observation key names,
image sizes, and state / action dimensions are all read from the
pretrained model's ``config.json`` — nothing is hardcoded.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.policies.act.modeling_act import ACTPolicy


class PolicyAgent:
    """Thin wrapper around a pretrained ACTPolicy for real-robot inference.

    Public API consumed by ``ros2_deploy.py`` and others::

        agent = PolicyAgent("/path/to/pretrained_model")
        action = agent.inference(obs)   # obs from ROS node
        agent.reset()                   # call on episode boundary
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.policy = self._load_policy()
        self.device = next(self.policy.parameters()).device

        self._image_keys: dict[str, tuple[int, int]] = {}
        self._state_key: str | None = None
        self._state_dim: int = 0
        self._action_dim: int = 0
        self._parse_model_config()

    # ── loading ────────────────────────────────────────────────────

    def _load_policy(self) -> ACTPolicy:
        print(f"[PolicyAgent] Loading model from: {self.model_path}")
        policy = ACTPolicy.from_pretrained(self.model_path)
        device = policy.config.device
        print(f"[PolicyAgent] Model loaded on {device}")
        return policy

    def _parse_model_config(self):
        """Extract image / state / action metadata from the loaded model config."""
        cfg = self.policy.config

        for key, feat in cfg.input_features.items():
            if feat.type is FeatureType.VISUAL:
                _, h, w = feat.shape  # (C, H, W)
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

    # ── public interface ───────────────────────────────────────────

    def inference(self, obs: dict) -> torch.Tensor:
        """Run one inference step.

        Args:
            obs: Observation dict from the ROS node with schema::

                    {
                        "images": {"<cam_name>": np.ndarray (H, W, 3) uint8},
                        "arm_gripper_joints": np.ndarray (state_dim,),
                    }

        Returns:
            Action tensor of shape ``(batch, action_dim)`` on the model
            device.  The caller typically does ``action[0].numpy()``.
        """
        batch = self._prepare_batch(obs)
        return self.policy.select_action(batch)

    def reset(self):
        """Reset the internal action queue.  Call on every episode start."""
        self.policy.reset()

    # ── observation preparation ────────────────────────────────────

    def _prepare_batch(self, obs: dict) -> dict[str, torch.Tensor]:
        """Convert a raw ROS observation dict into the batch format that
        ``ACTPolicy.select_action`` expects.

        Key mapping:
          obs["images"]["camera"]       →  "observation.images.camera"
          obs["arm_gripper_joints"]      →  "observation.state"
        """
        batch: dict[str, torch.Tensor] = {}

        for model_key, (target_w, target_h) in self._image_keys.items():
            # model_key is e.g. "observation.images.camera"
            # extract the short camera name after the last dot
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

