# -*- coding: utf-8 -*-
"""Convert HDF5 episode data to LeRobot Dataset V3.

All HDF5-to-feature mapping is driven by a JSON config file. See configs/ for examples.

Supported mapping features:
  - ``hdf5_key``  — read a single HDF5 dataset.
  - ``hdf5_keys`` — read and concatenate multiple HDF5 datasets along axis-1.
  - ``slice``     — ``[start, end]`` column slice applied to each key.
  - ``stats_override`` (top-level) — manually specify stats for any feature,
    overriding the auto-computed values after all episodes are saved.
"""

import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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


def _read_single_key(h5file: h5py.File, key: str, slc: list | None = None) -> np.ndarray:
    """Read one HDF5 dataset with optional column slice."""
    data = np.array(h5file[key], dtype=np.float32)
    if slc is not None:
        data = data[:, slc[0]:slc[1]]
    return data


def _read_numeric(h5file: h5py.File, mapping: dict) -> np.ndarray:
    """Read numeric field(s) from HDF5. Supports single key or concat of multiple keys."""
    if "hdf5_keys" in mapping:
        parts = []
        slices = mapping.get("slices", [None] * len(mapping["hdf5_keys"]))
        for key, slc in zip(mapping["hdf5_keys"], slices):
            parts.append(_read_single_key(h5file, key, slc))
        return np.concatenate(parts, axis=1)

    data = np.array(h5file[mapping["hdf5_key"]], dtype=np.float32)
    if "slice" in mapping:
        s = mapping["slice"]
        data = data[:, s[0]:s[1]]
    return data


def _decode_image_buffer(buf: np.ndarray, resize: tuple[int, int] | None) -> np.ndarray:
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if resize is not None:
        img = cv2.resize(img, resize)
    return img


def _read_images(
    h5file: h5py.File,
    mapping: dict,
    *,
    decode_workers: int = 1,
) -> np.ndarray:
    """Read and decode compressed images from HDF5, with optional resize.

    JPEG/PNG decode can run in parallel threads (decode_workers > 1). HDF5 bytes are
    read sequentially first to avoid concurrent access on the same file handle.
    """
    raw = h5file[mapping["hdf5_key"]]
    resize = tuple(mapping["resize"]) if "resize" in mapping else None
    buffers = [np.frombuffer(img_bytes, np.uint8) for img_bytes in raw]
    if decode_workers <= 1:
        images = [_decode_image_buffer(b, resize) for b in buffers]
    else:
        decoder = partial(_decode_image_buffer, resize=resize)
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            images = list(pool.map(decoder, buffers))
    return np.stack(images)


def process_episode(
    episode_path: Path,
    dataset: LeRobotDataset,
    task_name: str,
    mappings: list[dict],
    *,
    decode_workers: int = 1,
) -> bool:
    """Process a single HDF5 episode and add frames to dataset."""
    try:
        with h5py.File(episode_path, "r") as h5file:
            # Pre-load all mapped data
            loaded = {}
            for m in mappings:
                fkey = m["feature_key"]
                if m.get("decode") in ("jpeg", "png", "image"):
                    loaded[fkey] = _read_images(h5file, m, decode_workers=decode_workers)
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


def apply_stats_override(dataset_root: Path, overrides: dict) -> None:
    """Patch stats.json with manually specified values.

    ``overrides`` maps feature names to dicts of stat fields, e.g.::

        {
            "action": {
                "mean": [0.1, 0.2, ...],
                "std":  [1.0, 1.0, ...]
            }
        }

    Only the stat fields present in the override are replaced; the rest are
    kept from the auto-computed stats.
    """
    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        logging.warning(f"stats.json not found at {stats_path}, skipping override")
        return

    with open(stats_path, "r") as f:
        stats = json.load(f)

    for feature_key, override_fields in overrides.items():
        if feature_key not in stats:
            logging.warning(f"stats_override: feature '{feature_key}' not in stats.json, adding it")
            stats[feature_key] = {}
        for stat_name, value in override_fields.items():
            stats[feature_key][stat_name] = value
            logging.info(f"stats_override: {feature_key}.{stat_name} overridden ({len(value) if isinstance(value, list) else 1} values)")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    logging.info(f"stats.json updated with overrides at {stats_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="HDF5 to LeRobot Dataset Converter")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--repo_id", type=str, required=True, help="Dataset repository ID")
    parser.add_argument("--src_root", type=str, required=True, help="Source data directory")
    parser.add_argument("--tgt_path", type=str, required=True, help="Target output directory")
    parser.add_argument("--task_name", type=str, default="default_task", help="Task name identifier")
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=0,
        help=(
            "Thread count for parallel JPEG/PNG decode per episode (0 = auto: min(8, CPU count))."
            " Does not parallelize LeRobot dataset writes; use 1 if memory is tight."
        ),
    )
    parser.add_argument(
        "--stats-override",
        action="store_true",
        default=False,
        help="Apply stats_override from config JSON after conversion (default: off)",
    )
    args = parser.parse_args()

    decode_workers = args.decode_workers
    if decode_workers <= 0:
        decode_workers = min(8, (os.cpu_count() or 4))

    config = load_config(args.config)
    dataset = initialize_dataset(repo_id=args.repo_id, tgt_path=args.tgt_path, config=config)

    episode_rel = config.get("episode_path", "data/trajectory.hdf5")
    src_root = Path(args.src_root)
    episodes = sorted([ep for ep in src_root.iterdir() if ep.is_dir()])

    logging.info(
        f"Start processing {len(episodes)} episodes (decode workers per episode: {decode_workers})..."
    )
    for ep_dir in episodes:
        ep_path = ep_dir / episode_rel
        if process_episode(
            ep_path,
            dataset,
            args.task_name,
            config["mappings"],
            decode_workers=decode_workers,
        ):
            dataset.save_episode()
            logging.info(f"Saved episode: {ep_dir.name}")

    dataset.finalize()

    if args.stats_override:
        if "stats_override" in config:
            dataset_root = Path(args.tgt_path) / args.repo_id
            apply_stats_override(dataset_root, config["stats_override"])
        else:
            logging.warning("--stats-override flag set but no 'stats_override' section found in config")

    logging.info("Dataset conversion completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
