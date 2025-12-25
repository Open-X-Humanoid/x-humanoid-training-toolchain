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
import importlib

import gymnasium as gym
import numpy as np
from typing import Dict, Any

from lerobot.envs.configs import AlohaEnv, EnvConfig, PushtEnv, XarmEnv, MetaWorldEnv, LiberoEnv


class MetaWorldTaskWrapper(gym.Wrapper):
    """Wrapper to add task conditioning information to MetaWorld environments."""

    def __init__(self, env, task_name):
        super().__init__(env)
        self.task_name = task_name

    @property
    def task(self):
        """Task identifier for policy conditioning."""
        return self.task_name

    @property
    def task_description(self):
        """Human-readable task description."""
        return self.task_name

class MetaWorldImageWrapper(gym.ObservationWrapper):
    """Wrapper to add image observations to MetaWorld environments."""

    def __init__(self, env, image_size=(480, 480)):
        super().__init__(env)
        self.image_size = image_size

        # Update observation space to include image
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(*image_size, 3), dtype=np.uint8),
            'state': env.observation_space
        })

    def observation(self, obs):
        # Render image from the environment
        image = self.env.render(mode='rgb_array', resolution=self.image_size)
        return {'image': image, 'state': obs}

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self.observation(obs), reward, done, truncated, info


def resolve_and_update_env_config(
    cfg: EnvConfig, selected_episodes: list[int] | None = None, override_dataset_path: str | None = None
):
    """
    Updates the environment configuration with the data from the dataset if available.

    Args:
        cfg: The environment configuration to update.
        selected_episodes: If specified, this function will add the key `cfg.episode_data_index` which maps
            episode indices to their positions in the list. For example, if the episode indices in the
            dataset are [0, 1, 2, 3, 4] and you select [2, 4], then `cfg.episode_data_index` will be
            {0: 2, 1: 4}.
        override_dataset_path: If specified, this will override the dataset path in the configuration.

    Returns:
        The resolved environment configuration.
    """
    if override_dataset_path is not None:
        cfg.dataset_path = override_dataset_path

    # If there is no dataset associated with the environment, return the config as is.
    if not hasattr(cfg, 'dataset_path') or cfg.dataset_path is None:
        return cfg

    # dataset_info = ResolvedEnvInfo.from_hf_dataset(cfg.dataset_path)  # Commented out - import error
    # For now, skip dataset info resolution due to import issues

    if selected_episodes is not None:
        episode_data_index = {i: ep_idx for i, ep_idx in enumerate(selected_episodes)}
        cfg.episode_data_index = episode_data_index

    return cfg


def make_task(
    cfg: EnvConfig,
    n_envs: int = 1,
    use_async_envs: bool = True,
    return_unbatched_env: bool = False,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """
    Factory function to make envs, batched or unbatched.

    Args:
        cfg: The environment configuration to use.
        n_envs: The number of environments to create in the batched environment. If
            `return_unbatched_env` is True, this argument is ignored.
        use_async_envs: Whether to use asynchronous environments when batching. If `return_unbatched_env`
            is True, this argument is ignored.
        return_unbatched_env: If True, return the env directly without batching (ignores `n_envs`,
            `use_async_envs` arguments).

    Returns:
        A dictionary with the following structure:
        {
            suite_name: {
                task_id: env
            }
        }
        where the env is either a batched or unbatched environment.
    """
    # Resolve and update config from dataset info.
    cfg = resolve_and_update_env_config(cfg)

    # Determine the env class for vectorization
    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv

    # Create env functions based on env type
    if cfg.type == "metaworld":
        # Check for required metaworld import
        try:
            import metaworld
        except ImportError as e:
            raise ImportError(
                "MetaWorld is not installed. Please install it with:\n"
                "pip install metaworld"
            ) from e

        def make_metaworld_env():
            mt1 = metaworld.MT1(cfg.task)
            metaworld_kwargs = cfg.metaworld_kwargs or {}

            # Ensure camera_name is set for MT10
            if cfg.task.startswith("mt10-") and "camera_name" not in metaworld_kwargs:
                metaworld_kwargs["camera_name"] = "corner"

            # Initialize the base environment
            if hasattr(mt1, 'train_classes'):
                env_cls = mt1.train_classes[cfg.task]
            else:
                env_cls = list(mt1.train_tasks)[0].__class__

            env = env_cls(**metaworld_kwargs)
            env.set_task(mt1.train_tasks[0])

            # CRITICAL FIX: Add task conditioning wrapper
            env = MetaWorldTaskWrapper(env, cfg.task)

            # Add image wrapper only for visual observations
            if cfg.obs_type == "pixels":
                env = MetaWorldImageWrapper(env, image_size=(480, 480))
            # For state-only mode, no wrapper needed - raw observation is already the state vector

            return env

        env_fns = [make_metaworld_env for _ in range(n_envs)]

    elif cfg.type == "libero":
        # Handle Libero environments
        from lerobot.envs.libero import create_libero_envs

        return create_libero_envs(
            task=cfg.task,
            n_envs=n_envs,
            camera_name=cfg.camera_name,
            init_states=cfg.init_states,
            gym_kwargs=cfg.gym_kwargs,
            env_cls=env_cls,
        )

    else:
        # Handle standard gym environments
        package_name = f"gym_{cfg.type}"
        try:
            importlib.import_module(package_name)
        except ModuleNotFoundError as e:
            print(f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.type}]'`")
            raise e

        gym_handle = f"{package_name}/{cfg.task}"

        def _make_one():
            return gym.make(gym_handle, disable_env_checker=cfg.disable_env_checker, **(cfg.gym_kwargs or {}))

        env_fns = [_make_one for _ in range(n_envs)]

    # Create batched or unbatched environment
    if return_unbatched_env:
        # Return unbatched environment
        env = env_fns[0]()
        # normalize to {suite: {task_id: env}} for consistency
        return {cfg.type: {0: env}}
    else:
        # Return batched environment
        vec = env_cls(env_fns)

        # normalize to {suite: {task_id: vec_env}} for consistency
        return {cfg.type: {0: vec}}


def make_env(
    cfg: EnvConfig,
    n_envs: int = 1,
    use_async_envs: bool = True,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """
    Factory function to make batched envs.

    Note: Calls `make_task` with `return_unbatched_env=False`
    """
    return make_task(cfg, n_envs=n_envs, use_async_envs=use_async_envs, return_unbatched_env=False)


# Import ResolvedEnvInfo at the end to avoid circular import
# from lerobot.utils.hub import ResolvedEnvInfo  # Commented out - causing import error