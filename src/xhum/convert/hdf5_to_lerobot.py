# -*- coding: utf-8 -*-
"""Convert HDF5 episode data (RoboMIND format) to LeRobot Dataset V3."""

import argparse
import ast
import json
import logging
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_config(config_path: str) -> dict:
    """Load and process configuration file."""
    with open(config_path, "r") as f:
        features = json.load(f)

    # Convert shape fields from string to tuple if needed
    for field in ["action", "observation.state"]:
        if field in features and isinstance(features[field].get("shape"), str):
            features[field]["shape"] = ast.literal_eval(features[field]["shape"])

    logging.info(f"Loaded features config: {features}")
    return features


def initialize_dataset(
    repo_id: str, tgt_path: str, fps: int, robot_type: str, features: dict
) -> LeRobotDataset:
    """Initialize a LeRobot V3 dataset."""
    dataset_path = Path(tgt_path) / repo_id

    if dataset_path.exists():
        shutil.rmtree(dataset_path)
        logging.warning(f"Removed existing dataset: {dataset_path}")

    logging.info(f"Creating new dataset: {dataset_path}")
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=str(dataset_path),
        fps=fps,
        robot_type=robot_type,
        features=features,
    )


def process_episode(
    episode_path: Path, dataset: LeRobotDataset, task_name: str, image_size: tuple = (640, 360)
) -> bool:
    """Process a single HDF5 episode and add frames to dataset."""
    try:
        with h5py.File(episode_path, "r") as file:
            puppet_state = np.array(file["puppet/joint_position"])
            camera_top_rgb_images = [
                cv2.resize(cv2.imdecode(img_compressed, cv2.IMREAD_COLOR), image_size)
                for img_compressed in file["observations/rgb_images/camera_top"]
            ]
            rgb_camera_top = np.stack(camera_top_rgb_images)

    except (FileNotFoundError, OSError, KeyError) as e:
        logging.error(f"Skipped {episode_path}: {e}")
        return False

    num_frames = len(puppet_state)
    for i in tqdm(range(num_frames), desc=f"Processing {episode_path.name}"):
        frame_data = {
            "task": task_name,
            "action": puppet_state[i],
            "observation.state": puppet_state[i],
            "observation.images.camera_top": rgb_camera_top[i],
        }
        dataset.add_frame(frame_data)

    return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="HDF5 to LeRobot Dataset Converter")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--repo_id", type=str, required=True, help="Dataset repository ID")
    parser.add_argument("--src_root", type=str, required=True, help="Source data directory")
    parser.add_argument("--tgt_path", type=str, required=True, help="Target output directory")
    parser.add_argument("--task_name", type=str, default="default_task", help="Task name identifier")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--robot_type", type=str, default="tiangong", help="Robot type identifier")
    args = parser.parse_args()

    features = load_config(args.config)

    dataset = initialize_dataset(
        repo_id=args.repo_id,
        tgt_path=args.tgt_path,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
    )

    src_root = Path(args.src_root)
    episodes = sorted([ep for ep in src_root.iterdir() if ep.is_dir()])

    logging.info(f"Start processing {len(episodes)} episodes...")
    for ep_dir in episodes:
        ep_path = ep_dir / "data" / "trajectory.hdf5"
        if process_episode(ep_path, dataset, args.task_name):
            dataset.save_episode()
            logging.info(f"Saved episode: {ep_dir.name}")

    dataset.finalize()
    logging.info("Dataset conversion completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
