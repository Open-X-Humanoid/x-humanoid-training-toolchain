# -*- coding: utf-8 -*-
"""Convert HDF5 episode data to LeRobot Dataset V3.

All HDF5-to-feature mapping is driven by a JSON config file. See configs/ for examples.
"""

import argparse
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
    """Load conversion config (dataset meta + features + HDF5 mappings)."""
    with open(config_path, "r") as f:
        config = json.load(f)

    required = ["dataset", "features", "mappings"]
    for key in required:
        if key not in config:
            raise ValueError(f"Config missing required section: '{key}'")

    # Convert shape lists to tuples so they match numpy .shape comparisons
    for feat in config["features"].values():
        if isinstance(feat.get("shape"), list):
            feat["shape"] = tuple(feat["shape"])

    return config


def initialize_dataset(repo_id: str, tgt_path: str, config: dict) -> LeRobotDataset:
    """Initialize a LeRobot V3 dataset from config."""
    dataset_path = Path(tgt_path) / repo_id

    if dataset_path.exists():
        shutil.rmtree(dataset_path)
        logging.warning(f"Removed existing dataset: {dataset_path}")

    ds_cfg = config["dataset"]
    logging.info(f"Creating new dataset: {dataset_path}")
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=str(dataset_path),
        fps=ds_cfg["fps"],
        robot_type=ds_cfg.get("robot_type", "unknown"),
        features=config["features"],
    )


def _read_numeric(h5file: h5py.File, mapping: dict) -> np.ndarray:
    """Read a numeric (state/action) field from HDF5, with optional slicing."""
    data = np.array(h5file[mapping["hdf5_key"]], dtype=np.float32)
    if "slice" in mapping:
        s = mapping["slice"]
        data = data[:, s[0]:s[1]]
    return data


def _read_images(h5file: h5py.File, mapping: dict) -> np.ndarray:
    """Read and decode compressed images from HDF5, with optional resize."""
    raw = h5file[mapping["hdf5_key"]]
    resize = tuple(mapping["resize"]) if "resize" in mapping else None
    images = []
    for img_bytes in raw:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if resize is not None:
            img = cv2.resize(img, resize)
        images.append(img)
    return np.stack(images)


def process_episode(
    episode_path: Path,
    dataset: LeRobotDataset,
    task_name: str,
    mappings: list[dict],
) -> bool:
    """Process a single HDF5 episode and add frames to dataset."""
    try:
        with h5py.File(episode_path, "r") as h5file:
            # Pre-load all mapped data
            loaded = {}
            for m in mappings:
                fkey = m["feature_key"]
                if m.get("decode") in ("jpeg", "png", "image"):
                    loaded[fkey] = _read_images(h5file, m)
                else:
                    loaded[fkey] = _read_numeric(h5file, m)
    except (FileNotFoundError, OSError, KeyError) as e:
        logging.error(f"Skipped {episode_path}: {e}")
        return False

    # Use the first mapping's length as frame count (all should match)
    num_frames = len(next(iter(loaded.values())))
    for i in tqdm(range(num_frames), desc=f"Processing {episode_path.name}"):
        frame = {"task": task_name}
        for fkey, data in loaded.items():
            frame[fkey] = data[i]
        dataset.add_frame(frame)

    return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="HDF5 to LeRobot Dataset Converter")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--repo_id", type=str, required=True, help="Dataset repository ID")
    parser.add_argument("--src_root", type=str, required=True, help="Source data directory")
    parser.add_argument("--tgt_path", type=str, required=True, help="Target output directory")
    parser.add_argument("--task_name", type=str, default="default_task", help="Task name identifier")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = initialize_dataset(repo_id=args.repo_id, tgt_path=args.tgt_path, config=config)

    episode_rel = config.get("episode_path", "data/trajectory.hdf5")
    src_root = Path(args.src_root)
    episodes = sorted([ep for ep in src_root.iterdir() if ep.is_dir()])

    logging.info(f"Start processing {len(episodes)} episodes...")
    for ep_dir in episodes:
        ep_path = ep_dir / episode_rel
        if process_episode(ep_path, dataset, args.task_name, config["mappings"]):
            dataset.save_episode()
            logging.info(f"Saved episode: {ep_dir.name}")

    dataset.finalize()
    logging.info("Dataset conversion completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
