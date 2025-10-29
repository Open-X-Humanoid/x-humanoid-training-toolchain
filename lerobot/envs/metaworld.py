#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MetaWorld expert models for generating demonstration data.

This module provides expert models that can generate high-quality trajectories
for MetaWorld tasks using the built-in oracle policies.

Usage:
    1. Install: pip install -e ".[metaworld]"
    2. Apply patch: bash patches/metaworld/apply_patches.sh
    3. Generate data: python lerobot/datasets/generate_metaworld_data.py --task=reach-v3 --num_episodes=100
    4. Train: python lerobot/scripts/lerobot_train.py --dataset.repo_id=local/your_dataset --env.type=metaworld --env.task=reach-v3
    5. Evaluate: python lerobot/scripts/lerobot_eval.py --policy.path=outputs/.../checkpoints/last --env.type=metaworld --env.task=reach-v3
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ExpertModel(ABC):
    """Abstract base class for expert trajectory generators."""

    def __init__(self, device: str = "cuda"):
        """Initialize the expert model.

        Args:
            device: Device to run the model on ("cuda" or "cpu")
        """
        self.device = device
        self.initialized = False

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the model with any necessary resources."""
        pass

    @abstractmethod
    def generate_action(
        self, observation: np.ndarray, task_description: Optional[str] = None, **kwargs
    ) -> np.ndarray:
        """Generate an expert action given an observation.

        Args:
            observation: Current environment observation
            task_description: Optional natural language task description
            **kwargs: Additional model-specific arguments

        Returns:
            Action to take in the environment
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state of the expert."""
        pass

    def generate_trajectory(
        self, env, max_steps: int = 500, task_description: Optional[str] = None, render: bool = False
    ) -> Tuple[bool, Dict[str, np.ndarray]]:
        """Generate a complete trajectory in an environment.

        Args:
            env: Gymnasium environment
            max_steps: Maximum steps in the trajectory
            task_description: Optional task description
            render: Whether to render the environment

        Returns:
            Tuple of (success, trajectory_dict) where trajectory_dict contains
            observations, actions, rewards, etc.
        """
        observations = []
        actions = []
        rewards = []
        dones = []
        infos = []

        obs, info = env.reset()
        self.reset()

        for step in range(max_steps):
            # Generate expert action
            action = self.generate_action(obs, task_description)

            # Store trajectory data
            observations.append(obs)
            actions.append(action)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)

            if render:
                env.render()

            if terminated or truncated:
                break

            obs = next_obs

        # Check if task was successful
        success = info.get("success", False) if info else False

        trajectory = {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "dones": np.array(dones),
            "success": success,
            "infos": infos,
        }

        return success, trajectory


class MetaWorldOracleExpert(ExpertModel):
    """Expert model using MetaWorld's built-in oracle policies.

    These are the official scripted policies provided by MetaWorld that
    achieve near-perfect performance on their respective tasks.
    """

    def __init__(self, task_name: str, **kwargs):
        """Initialize MetaWorld oracle expert.

        Args:
            task_name: Name of the MetaWorld task (e.g., 'reach-v3')
        """
        super().__init__(task_name, **kwargs)
        self.task_name = task_name
        self.policy = None
        self._policy_map = None

    def _get_policy_map(self):
        """Lazy load policy map to avoid import errors."""
        if self._policy_map is None:
            from metaworld.policies import (
                SawyerButtonPressTopdownV3Policy,
                SawyerButtonPressV3Policy,
                SawyerDoorOpenV3Policy,
                SawyerDrawerCloseV3Policy,
                SawyerDrawerOpenV3Policy,
                SawyerPegInsertionSideV3Policy,
                SawyerPickPlaceV3Policy,
                SawyerPushV3Policy,
                SawyerReachV3Policy,
                SawyerWindowCloseV3Policy,
                SawyerWindowOpenV3Policy,
            )

            self._policy_map = {
                "reach-v3": SawyerReachV3Policy,
                "push-v3": SawyerPushV3Policy,
                "pick-place-v3": SawyerPickPlaceV3Policy,
                "door-open-v3": SawyerDoorOpenV3Policy,
                "drawer-open-v3": SawyerDrawerOpenV3Policy,
                "drawer-close-v3": SawyerDrawerCloseV3Policy,
                "button-press-topdown-v3": SawyerButtonPressTopdownV3Policy,
                "button-press-v3": SawyerButtonPressV3Policy,
                "window-open-v3": SawyerWindowOpenV3Policy,
                "window-close-v3": SawyerWindowCloseV3Policy,
                "peg-insert-side-v3": SawyerPegInsertionSideV3Policy,
            }
        return self._policy_map

    def initialize(self, env: Any) -> None:
        """Initialize the oracle policy.

        Args:
            env: MetaWorld environment instance
        """
        policy_map = self._get_policy_map()

        if self.task_name not in policy_map:
            available_tasks = list(policy_map.keys())
            raise ValueError(
                f"No oracle policy available for task '{self.task_name}'. "
                f"Available tasks: {available_tasks}"
            )

        policy_class = policy_map[self.task_name]
        self.policy = policy_class()

        logger.info(f"Initialized MetaWorld oracle expert for {self.task_name}")
        logger.info(f"Using policy: {policy_class.__name__}")

    def generate_action(self, observation: np.ndarray) -> np.ndarray:
        """Generate expert action using MetaWorld's oracle policy.

        Args:
            observation: Current environment observation

        Returns:
            Expert action
        """
        if self.policy is None:
            raise RuntimeError("Policy not initialized. Call initialize() first.")

        # MetaWorld policies expect 39-dimensional observations
        if isinstance(observation, dict):
            # Handle dict observations from wrapped environments
            if "state" in observation:
                obs = observation["state"]
            else:
                obs = observation.get("observation", observation)
        else:
            obs = observation

        # Ensure observation is the right shape
        if obs.shape[0] != 39:
            raise ValueError(f"Expected 39-dim observation, got {obs.shape}")

        # Get action from oracle policy
        action = self.policy.get_action(obs)

        # MetaWorld policies return 4D actions, but environments expect 4D
        if action.shape[0] > 4:
            action = action[:4]

        return action

    def generate_trajectory(
        self, env: Any, max_steps: int = 200, render: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """Generate a complete trajectory using the oracle policy.

        Args:
            env: MetaWorld environment
            max_steps: Maximum steps per episode
            render: Whether to render the environment

        Returns:
            Tuple of (success, trajectory_dict)
        """
        if self.policy is None:
            self.initialize(env)

        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        obs, _ = env.reset()
        success = False

        for step in range(max_steps):
            # Store observation
            observations.append(obs.copy() if isinstance(obs, np.ndarray) else obs)

            # Generate action
            action = self.generate_action(obs)
            actions.append(action.copy())

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            next_observations.append(next_obs.copy() if isinstance(next_obs, np.ndarray) else next_obs)
            done = terminated or truncated
            dones.append(done)

            # Check success
            if info.get("success", 0.0) > 0:
                success = True

            # Render if requested
            if render:
                env.render()

            obs = next_obs

            if done:
                break

        trajectory = {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "next_observations": np.array(next_observations),
            "dones": np.array(dones),
            "success": success,
            "length": len(observations),
        }

        return success, trajectory

    def reset(self) -> None:
        """Reset the expert (no internal state for oracle policies)."""
        pass  # Oracle policies are stateless
