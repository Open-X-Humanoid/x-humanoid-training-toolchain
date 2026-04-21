#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ACT policy inference (LeRobot v0.5.1) — runs under Python 3.12+ only.

Lives under ``src/xhum/deploy_decouple/policy`` so ROS (Python 3.10) never
imports LeRobot here. Keep in sync with ``src/xhum/deploy/policy_agent.py``
when the model I/O contract changes.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


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
        self._warned_cam_remap = False

        self.preprocessor: Any = None
        self.postprocessor: Any = None
        self._load_pre_post_processors()

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

    def _load_pre_post_processors(self) -> None:
        """Match LeRobot ``predict_action`` / async ``policy_server``: normalize obs, denorm action.

        If ``policy_preprocessor.json`` / ``policy_postprocessor.json`` are absent (legacy export),
        skip processors and call ``select_action`` on the raw batch (previous behavior).
        """
        pre_json = self.model_path / "policy_preprocessor.json"
        post_json = self.model_path / "policy_postprocessor.json"
        if not (pre_json.is_file() and post_json.is_file()):
            print(
                "[PolicyAgent] No policy_preprocessor.json / policy_postprocessor.json next to weights; "
                "inference uses raw ``select_action`` (no dataset normalize / no action denorm).",
                flush=True,
            )
            return
        dev = str(self.device)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.model_path),
            preprocessor_overrides={
                "device_processor": {"device": dev, "float_dtype": None},
                "rename_observations_processor": {"rename_map": {}},
            },
            postprocessor_overrides={"device_processor": {"device": dev, "float_dtype": None}},
        )
        print(
            "[PolicyAgent] Loaded LeRobot preprocessor + postprocessor "
            "(obs normalize like training, action denorm before return).",
            flush=True,
        )

    def inference(self, obs: dict) -> torch.Tensor:
        batch = self._prepare_batch(obs)
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        action = self.policy.select_action(batch)
        if self.postprocessor is not None:
            action = self.postprocessor(action)
        return action

    def reset(self):
        self.policy.reset()
        if self.preprocessor is not None:
            self.preprocessor.reset()
        if self.postprocessor is not None:
            self.postprocessor.reset()

    def _prepare_batch(self, obs: dict) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}
        imgs = obs.get("images")
        if not isinstance(imgs, dict) or not imgs:
            raise ValueError("obs must have non-empty dict obs['images'] (short_name -> uint8 RGB HWC)")

        # One visual input + one tensor: allow HDF5 key (e.g. camera_head) vs checkpoint (camera).
        remap_one: str | None = None
        if len(self._image_keys) == 1 and len(imgs) == 1:
            only_key = next(iter(imgs))
            model_key0 = next(iter(self._image_keys))
            expected = model_key0.rsplit(".", maxsplit=1)[-1]
            if only_key != expected:
                remap_one = only_key

        for model_key, (target_w, target_h) in self._image_keys.items():
            cam_name = model_key.rsplit(".", maxsplit=1)[-1]
            if cam_name in imgs:
                img = imgs[cam_name]
            elif remap_one is not None and remap_one in imgs:
                img = imgs[remap_one]
                if not self._warned_cam_remap:
                    print(
                        f"[PolicyAgent] Using obs['images'][{remap_one!r}] for {model_key!r} "
                        f"(checkpoint expects short name {cam_name!r}). "
                        "Set obs_camera_key in deploy YAML to match the model to silence this.",
                        flush=True,
                    )
                    self._warned_cam_remap = True
            else:
                need = ", ".join(k.rsplit(".", maxsplit=1)[-1] for k in self._image_keys)
                avail = ", ".join(sorted(imgs.keys()))
                raise ValueError(
                    f"obs['images'] missing {cam_name!r} (model expects: {need}); got keys: [{avail}]. "
                    "Set obs_camera_key to the same short name as in config.input_features "
                    "(see pretrained_model/policy_preprocessor.json)."
                )
            img = cv2.resize(img, dsize=(target_w, target_h))
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            batch[model_key] = img_t.unsqueeze(0).to(self.device, non_blocking=True)

        if self._state_key is not None:
            state = obs["arm_gripper_joints"]
            state_t = torch.from_numpy(np.asarray(state, dtype=np.float32))
            batch[self._state_key] = state_t.unsqueeze(0).to(self.device, non_blocking=True)

        return batch
