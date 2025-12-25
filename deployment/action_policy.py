#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
from pathlib import Path

# Third-party imports
import cv2
import numpy as np
import torch

# Local imports
from lerobot.policies.act.modeling_act import ACTPolicy as Policy


class PolicyAgent:
    """Agent class for handling policy model inference.
    
    This class manages the loading and execution of a pre-trained policy model,
    handling both real and simulated observation data for inference. It provides
    functionality for model loading, device selection, and observation processing.
    """
    
    def __init__(self, model_path):
        """Initialize the policy agent.
        
        Args:
            model_path (str): Path to the pre-trained model file.
        """
        self.model_path = Path(model_path)
        self.device = self._get_device()
        self.policy = self._load_policy()
        self.cnt = 0
    
    def _load_policy(self):
        """Load pretrained model and set it to specified device.

        Returns:
            Policy: Loaded and configured policy model.
        """
        print(f"Loading model from: {self.model_path}")
        policy = Policy.from_pretrained(self.model_path)
        policy.eval()
        policy.to(self.device)
        return policy
    
    def _get_device(self):
        """Determine and return the available computing device.
        
        Returns:
            torch.device: The selected device (GPU if available, otherwise CPU).
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available. Device set to:", device)
        else:
            device = torch.device("cpu")
            print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
        return device

    def inference(self, obs):
        """Perform model inference on observation data.

        Args:
            obs (dict, optional): Observation data containing images and joint positions.
                                If not provided, simulated data will be generated.

        Returns:
            dict: Model inference results containing predicted actions.
        """
        if obs is None:
            print("Using simulated observation data")
            obs = self.generate_obs()
        
        input_data = self.prepare_inference_obs(obs)
        # print("input_data",input_data)
        return self.policy.select_action(input_data)
    
    def generate_obs(self):
        """Generate simulated observation data for testing.
        
        Returns:
            dict: Simulated observation data containing:
                - images: Dict with camera images
                - qpos: Random joint positions
                - arm_gripper_joints: Random gripper joint positions
        """
        obs = {
            'images': {
                'camera': cv2.imencode('.jpg', np.random.randn(360, 640, 3))[1],
            },
            'qpos': np.random.randn(8,),
            'arm_gripper_joints': np.random.randn(16,),
        }
        return obs
    
    def reset(self):
        """Reset the policy model's internal state."""
        self.policy.reset()

    def prepare_inference_obs(self, obs):
        """Process and prepare observation data for model inference.
        
        Args:
            obs (dict): Raw observation data containing images and joint positions.
        
        Returns:
            dict: Processed observation data ready for model inference, including:
                - Normalized and resized camera images as tensors
                - Joint position data as tensors
        """
        inference_data = {}
        camera_names = ['camera']

        # Process image data
        for cam_name in camera_names:
            cam_img = obs['images'][cam_name]
            cam_img = cv2.resize(cam_img, dsize=(640, 360))
            
            cam_img_tensor = torch.from_numpy(cam_img).permute(2, 0, 1).float() / 255.0
            inference_data[f'observation.images.camera_{cam_name}'] = (
                cam_img_tensor.unsqueeze(0).to(self.device, non_blocking=True)
            )
            # #TODO fixed name
            # inference_data[f'observation.images.camera_top'] = (
            #     cam_img_tensor.unsqueeze(0).to(self.device, non_blocking=True)
            # )

        self.cnt += 1

        # Process joint position data
        qpos = obs['arm_gripper_joints']
        qpos_data = torch.from_numpy(qpos).float()
        inference_data['observation.state'] = qpos_data.unsqueeze(0).to(self.device, non_blocking=True)

        return inference_data