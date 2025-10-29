#!/usr/bin/env python
"""Generate expert demonstration data for MetaWorld tasks.

This script uses expert models (scripted or learned) to generate high-quality
trajectory data that can be used for imitation learning.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
import gymnasium as gym
import metaworld

from lerobot.envs.metaworld import MetaWorldOracleExpert
from lerobot.envs.factory import make_env
from lerobot.envs.configs import MetaWorldEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate expert demonstration data for MetaWorld tasks"
    )
    
    # Task configuration
    parser.add_argument(
        "--task",
        type=str,
        default="reach-v3",
        help="MetaWorld task name (e.g., reach-v3, push-v3, pick-place-v3)"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to generate"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per episode"
    )
    
    # Expert model configuration
    # Using MetaWorld built-in oracle policies by default
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/expert_data",
        help="Directory to save generated data"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name for the dataset (default: expert_{task}_{timestamp})"
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="lerobot",
        choices=["lerobot", "numpy", "hdf5"],
        help="Format to save the data"
    )
    
    # Environment configuration
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes during generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run expert model on"
    )
    
    # Quality control
    parser.add_argument(
        "--min_success_rate",
        type=float,
        default=0.8,
        help="Minimum success rate required (will retry if below)"
    )
    parser.add_argument(
        "--filter_failures",
        action="store_true",
        help="Only save successful trajectories"
    )
    
    return parser.parse_args()


def create_metaworld_env(task_name: str) -> gym.Env:
    """Create a MetaWorld environment."""
    mt1 = metaworld.MT1(task_name)
    env_cls = mt1.train_classes[task_name]
    env = env_cls(render_mode="rgb_array" if args.render else None)
    env.set_task(mt1.train_tasks[0])
    return env


def generate_expert_trajectories(
    env: gym.Env,
    expert: "ExpertModel",
    num_episodes: int,
    max_steps: int,
    render: bool = False,
    filter_failures: bool = False
) -> List[Dict]:
    """Generate expert trajectories using the given expert model.
    
    Returns:
        List of trajectory dictionaries
    """
    trajectories = []
    successes = 0
    
    pbar = tqdm(total=num_episodes, desc="Generating trajectories")
    
    episode = 0
    while episode < num_episodes:
        # Generate trajectory
        success, trajectory = expert.generate_trajectory(
            env,
            max_steps=max_steps,
            render=render
        )
        
        if success:
            successes += 1
        
        # Filter out failures if requested
        if filter_failures and not success:
            logger.info(f"Episode {episode}: Failed, skipping...")
            continue
        
        # Add metadata
        trajectory["episode_id"] = episode
        trajectory["task"] = args.task
        trajectory["expert_type"] = "oracle"  # MetaWorld uses oracle expert
        trajectory["timestamp"] = datetime.now().isoformat()
        
        trajectories.append(trajectory)
        
        # Update progress
        pbar.update(1)
        pbar.set_postfix({
            "success_rate": f"{successes/(episode+1)*100:.1f}%",
            "saved": len(trajectories)
        })
        
        episode += 1
    
    pbar.close()
    
    # Report statistics
    total_success_rate = successes / num_episodes
    logger.info(f"Generated {len(trajectories)} trajectories")
    logger.info(f"Overall success rate: {total_success_rate*100:.1f}%")
    
    if total_success_rate < args.min_success_rate:
        logger.warning(
            f"Success rate {total_success_rate*100:.1f}% is below minimum "
            f"{args.min_success_rate*100:.1f}%"
        )
    
    return trajectories


def save_trajectories_lerobot(
    trajectories: List[Dict],
    output_dir: Path,
    dataset_name: str,
    task_name: str
) -> None:
    """Save trajectories in LeRobot v3.0 format with minimal fields."""
    import pandas as pd
    from datasets import Dataset
    
    from lerobot.datasets.utils import (
        write_info, write_episodes, write_stats, write_tasks, 
        serialize_dict, DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH
    )
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
    
    # Prepare data for HuggingFace dataset - all required fields for v3.0 format
    data_dict = {
        "observation.state": [],
        "observation.environment_state": [],
        "task": [],  # CRITICAL: Add 'task' key to match evaluation environment
        "action": [],
        "episode_index": [],
        "frame_index": [],
        "timestamp": [],
        "next.done": [],
        "next.success": [],
        "next.reward": [],
        "task_index": [],
    }
    
    # Prepare episode metadata and stats for v3.0 format
    episodes_data = []
    episodes_stats = {}
    
    episode_idx = 0
    total_frames = 0
    
    for traj in trajectories:
        obs = traj["observations"]
        actions = traj["actions"]
        rewards = traj["rewards"] 
        dones = traj["dones"]
        success = traj["success"]
        
        episode_length = len(actions)
        episode_obs_states = []
        episode_env_states = []
        episode_actions = []
        
        for i in range(episode_length):
            # Extract observation components
            if len(obs[i]) >= 39:
                state = obs[i][:4].astype(np.float32)  # End-effector state
                env_state = obs[i].astype(np.float32)  # Full state
            else:
                state = obs[i][:4].astype(np.float32) if len(obs[i]) >= 4 else np.zeros(4, dtype=np.float32)
                env_state = np.pad(obs[i], (0, 39 - len(obs[i]))).astype(np.float32)
            
            # CRITICAL: Clip actions to [-1, 1] range for environment compatibility
            clipped_action = np.clip(actions[i].astype(np.float32), -1.0, 1.0)
            
            # CRITICAL: Add 'task' observation to match evaluation environment  
            # MetaWorld reach-v3 task representation (single task index)
            task_obs = np.float32(0.0)  # Task identifier for reach-v3 as scalar
            
            data_dict["observation.state"].append(state)
            data_dict["observation.environment_state"].append(env_state)
            data_dict["task"].append(task_obs)
            data_dict["action"].append(clipped_action)
            data_dict["episode_index"].append(episode_idx)
            data_dict["frame_index"].append(total_frames)
            data_dict["timestamp"].append(float(i) / 50.0)  # Assuming 50Hz
            
            # Add next step information
            is_last_step = (i == episode_length - 1)
            data_dict["next.done"].append(dones[i] if i < len(dones) else is_last_step)
            data_dict["next.success"].append(success if is_last_step else False)
            data_dict["next.reward"].append(float(rewards[i]) if i < len(rewards) else 0.0)
            
            # Add task index (always 0 for our single task)
            data_dict["task_index"].append(0)
            
            episode_obs_states.append(state)
            episode_env_states.append(env_state)
            episode_actions.append(actions[i])
            total_frames += 1
        
        # Create episode metadata for v3.0 format
        episode_start_idx = total_frames - episode_length  # Start index in dataset
        episode_end_idx = total_frames  # End index in dataset
        
        episodes_data.append({
            "episode_index": episode_idx,
            "length": episode_length,
            "tasks": [task_name],
            "data_chunk_index": 0,
            "data_file_index": 0,
            "dataset_from_index": episode_start_idx,
            "dataset_to_index": episode_end_idx,
        })
        
        # Compute episode-level statistics
        episode_stats = {
            "observation.state": {
                "mean": np.array(episode_obs_states).mean(axis=0),
                "std": np.array(episode_obs_states).std(axis=0),
                "min": np.array(episode_obs_states).min(axis=0),
                "max": np.array(episode_obs_states).max(axis=0),
                "count": np.array([episode_length])
            },
            "observation.environment_state": {
                "mean": np.array(episode_env_states).mean(axis=0),
                "std": np.array(episode_env_states).std(axis=0),
                "min": np.array(episode_env_states).min(axis=0),
                "max": np.array(episode_env_states).max(axis=0),
                "count": np.array([episode_length])
            },
            "action": {
                "mean": np.array(episode_actions).mean(axis=0),
                "std": np.array(episode_actions).std(axis=0),
                "min": np.array(episode_actions).min(axis=0),
                "max": np.array(episode_actions).max(axis=0),
                "count": np.array([episode_length])
            }
        }
        episodes_stats[episode_idx] = episode_stats
        episode_idx += 1
    
    # Convert to properly typed lists for HuggingFace Dataset
    data_dict["observation.state"] = [s.tolist() for s in data_dict["observation.state"]]
    data_dict["observation.environment_state"] = [s.tolist() for s in data_dict["observation.environment_state"]]
    # task is already scalar, no conversion needed
    data_dict["action"] = [s.tolist() for s in data_dict["action"]]
    # Keep other fields as they are - they have the correct types already
    
    # Create HuggingFace dataset for main data
    dataset = Dataset.from_dict(data_dict)
    
    # Save dataset
    dataset_path = output_dir / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Create data directory and save dataset there
    data_dir = dataset_path / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Save data as parquet (v3.0 format)
    data_file_path = data_dir / "chunk-000" / "episode_000000.parquet"
    data_file_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(data_file_path))
    
    # Create info.json for v3.0 format - include ALL fields present in parquet files
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": 50,
        "env": task_name,
        "robot_type": "metaworld",
        "features": {
            "observation.state": {"shape": [4], "dtype": "float32"},
            "observation.environment_state": {"shape": [39], "dtype": "float32"},
            "task": {"shape": [1], "dtype": "float32"},  # CRITICAL: Add task feature (scalar as [1])
            "action": {"shape": [4], "dtype": "float32"},
            "episode_index": {"shape": [1], "dtype": "int64"},
            "frame_index": {"shape": [1], "dtype": "int64"},
            "timestamp": {"shape": [1], "dtype": "float64"},
            "next.done": {"shape": [1], "dtype": "bool"},
            "next.success": {"shape": [1], "dtype": "bool"},
            "next.reward": {"shape": [1], "dtype": "float64"},
            "task_index": {"shape": [1], "dtype": "int64"}
        },
        "total_episodes": len(trajectories),
        "total_frames": total_frames,
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH,
        "splits": {"train": f"0:{len(trajectories)}"},
        "version": "1.0.0"
    }
    
    # Use v3.0 utilities to write metadata
    write_info(info, dataset_path)
    
    # Create and write episodes metadata using v3.0 format
    episodes_dataset = Dataset.from_list(episodes_data)
    write_episodes(episodes_dataset, dataset_path)
    
    # Create and write tasks using v3.0 format
    tasks_df = pd.DataFrame([{"task": task_name, "task_index": 0}])
    write_tasks(tasks_df, dataset_path)
    
    # Aggregate and write statistics using v3.0 format
    aggregated_stats = aggregate_stats(list(episodes_stats.values()))
    write_stats(aggregated_stats, dataset_path)
    
    logger.info(f"Saved dataset to {dataset_path}")
    logger.info(f"Dataset info: {info}")


def save_trajectories_numpy(
    trajectories: List[Dict],
    output_dir: Path,
    dataset_name: str
) -> None:
    """Save trajectories as numpy arrays."""
    save_dir = output_dir / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, traj in enumerate(trajectories):
        episode_dir = save_dir / f"episode_{i:04d}"
        episode_dir.mkdir(exist_ok=True)
        
        # Save each component
        np.save(episode_dir / "observations.npy", traj["observations"])
        np.save(episode_dir / "actions.npy", traj["actions"])
        np.save(episode_dir / "rewards.npy", traj["rewards"])
        np.save(episode_dir / "dones.npy", traj["dones"])
        
        # Save metadata
        metadata = {
            "success": bool(traj["success"]),
            "task": traj["task"],
            "expert_type": traj["expert_type"],
            "timestamp": traj["timestamp"],
            "episode_length": len(traj["actions"])
        }
        
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {len(trajectories)} episodes to {save_dir}")


def main(args):
    """Main function."""
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset name if not provided
    if args.dataset_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.dataset_name = f"expert_{args.task.replace('-', '_')}_{timestamp}"
    
    # Create environment
    logger.info(f"Creating MetaWorld environment: {args.task}")
    env = create_metaworld_env(args.task)
    
    # Create expert model using MetaWorld oracle
    logger.info(f"Creating MetaWorld oracle expert model")
    expert = MetaWorldOracleExpert(task_name=args.task)
    
    expert.initialize(env)
    
    # Generate trajectories
    logger.info(f"Generating {args.num_episodes} expert trajectories...")
    trajectories = generate_expert_trajectories(
        env,
        expert,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render=args.render,
        filter_failures=args.filter_failures
    )
    
    if len(trajectories) == 0:
        logger.error("No trajectories generated!")
        return
    
    # Save trajectories
    logger.info(f"Saving trajectories in {args.save_format} format...")
    if args.save_format == "lerobot":
        save_trajectories_lerobot(
            trajectories,
            output_dir,
            args.dataset_name,
            args.task
        )
    elif args.save_format == "numpy":
        save_trajectories_numpy(trajectories, output_dir, args.dataset_name)
    else:
        raise ValueError(f"Unknown save format: {args.save_format}")
    
    logger.info("Data generation complete!")
    
    # Print summary
    print("\n" + "="*50)
    print("GENERATION SUMMARY")
    print("="*50)
    print(f"Task: {args.task}")
    print(f"Episodes generated: {len(trajectories)}")
    print(f"Success rate: {sum(t['success'] for t in trajectories) / len(trajectories) * 100:.1f}%")
    print(f"Data saved to: {output_dir / args.dataset_name}")
    print("="*50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
