#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Policy inference wrapper for TienKung robot deployment."""

from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy


class PolicyAgent:
    """Agent class for handling policy model inference."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.device = self._get_device()
        self.policy = self._load_policy()
        self.cnt = 0

    def _load_policy(self):
        print(f"Loading model from: {self.model_path}")
        policy = ACTPolicy.from_pretrained(self.model_path)
        policy.eval()
        policy.to(self.device)
        return policy

    def _get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available. Device set to:", device)
        else:
            device = torch.device("cpu")
            print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
        return device

    def inference(self, obs):
        if obs is None:
            print("Using simulated observation data")
            obs = self.generate_obs()

        input_data = self.prepare_inference_obs(obs)
        return self.policy.select_action(input_data)

    def generate_obs(self):
        return {
            "images": {
                "camera": cv2.imencode(".jpg", np.random.randn(360, 640, 3))[1],
            },
            "qpos": np.random.randn(8),
            "arm_gripper_joints": np.random.randn(16),
        }

    def reset(self):
        self.policy.reset()

    def prepare_inference_obs(self, obs):
        inference_data = {}
        camera_names = ["camera"]

        for cam_name in camera_names:
            cam_img = obs["images"][cam_name]
            cam_img = cv2.resize(cam_img, dsize=(640, 360))
            cam_img_tensor = torch.from_numpy(cam_img).permute(2, 0, 1).float() / 255.0
            inference_data[f"observation.images.camera_{cam_name}"] = (
                cam_img_tensor.unsqueeze(0).to(self.device, non_blocking=True)
            )

        self.cnt += 1

        qpos = obs["arm_gripper_joints"]
        qpos_data = torch.from_numpy(qpos).float()
        inference_data["observation.state"] = qpos_data.unsqueeze(0).to(self.device, non_blocking=True)

        return inference_data
