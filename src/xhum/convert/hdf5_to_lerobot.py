# -*- coding: utf-8 -*-
"""Convert HDF5 episode data to LeRobot Dataset V3.

All HDF5-to-feature mapping is driven by a JSON config file. See configs/ for examples.

Supported mapping features:
  - ``hdf5_key``  — read a single HDF5 dataset.
  - ``hdf5_keys`` — read and concatenate multiple HDF5 datasets along axis-1.
  - ``slice``     — ``[start, end]`` column slice applied to each key.
  - ``divide_by`` — scalar divisor applied after read (per-key in ``hdf5_keys`` lists).
  - ``stats_override`` (top-level) — manually specify stats for any feature,
    overriding the auto-computed values after all episodes are saved.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
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


def initialize_dataset(
    repo_id: str,
    tgt_path: str,
    config: dict,
    *,
    streaming_encoding: bool = True,
    encoder_queue_maxsize: int = 1024,
) -> LeRobotDataset:
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
        # With streaming encoding each decoded frame goes straight to the video
        # encoder instead of being buffered as a temporary PNG and read back. Frame
        # data and encoded video come out byte-identical; only the image statistics
        # differ slightly, since the streaming encoder accumulates them over every
        # frame while the default path samples a subset.
        streaming_encoding=streaming_encoding,
        # LeRobot's streaming encoder drops a frame once its queue has been full for
        # 100ms rather than blocking, which suits live capture but silently loses data
        # offline: the default queue of 30 lost 2 of 72649 frames on one camera here.
        # The queue is rebuilt per episode, so one that can hold the longest episode
        # never fills and the drop path is unreachable. See plan_encoder_queue().
        encoder_queue_maxsize=encoder_queue_maxsize,
    )


def _read_single_key(h5file: h5py.File, key: str, slc: list | None = None) -> np.ndarray:
    """Read one HDF5 dataset with optional column slice."""
    data = np.array(h5file[key], dtype=np.float32)
    if slc is not None:
        data = data[:, slc[0]:slc[1]]
    return data


def _apply_divide(data: np.ndarray, divisor: float | None) -> np.ndarray:
    if divisor is not None:
        data = data / np.float32(divisor)
    return data


def _read_numeric(h5file: h5py.File, mapping: dict) -> np.ndarray:
    """Read numeric field(s) from HDF5. Supports single key or concat of multiple keys."""
    if "hdf5_keys" in mapping:
        keys = mapping["hdf5_keys"]
        n = len(keys)
        parts = []
        slices = mapping.get("slices", [None] * n)
        divisors = mapping.get("divide_by", [None] * n)
        if len(slices) != n:
            raise ValueError(f"slices length {len(slices)} != hdf5_keys length {n}")
        if len(divisors) != n:
            raise ValueError(f"divide_by length {len(divisors)} != hdf5_keys length {n}")
        for key, slc, divisor in zip(keys, slices, divisors):
            parts.append(_apply_divide(_read_single_key(h5file, key, slc), divisor))
        return np.concatenate(parts, axis=1)

    data = np.array(h5file[mapping["hdf5_key"]], dtype=np.float32)
    if "slice" in mapping:
        s = mapping["slice"]
        data = data[:, s[0]:s[1]]
    return _apply_divide(data, mapping.get("divide_by"))


def _decode_image_buffer(buf: np.ndarray, resize: tuple[int, int] | None) -> np.ndarray:
    # cv2.imdecode 总是返回 BGR；LeRobot 视频编码端期望 RGB，否则颜色红蓝互换。
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def _episode_frame_count(h5file, mappings: list[dict]) -> int:
    """Frames this episode will feed to the encoder.

    Mirrors ``process_episode``, which takes its frame count from the first mapping's
    loaded array. Reading the dataset's shape touches metadata only, so this stays cheap.
    """
    first = mappings[0]
    keys = first.get("hdf5_keys") or [first.get("hdf5_key")]
    key = keys[0]
    if not key or key not in h5file:
        return 0
    return int(h5file[key].shape[0])


def plan_encoder_queue(episodes: list[Path], episode_rel: str, mappings: list[dict], cap: int = 4096) -> int:
    """Pick an encoder queue size that makes LeRobot's frame-drop path unreachable.

    The streaming encoder drops a frame whenever its queue is full for more than 100ms,
    and rebuilds that queue for every episode. A queue that can hold the longest episode
    therefore never fills, so no frame is ever dropped. Draining it happens concurrently
    with feeding and stays well inside the encoder's 120s join budget at realistic
    encode rates; ``cap`` keeps a pathological episode from blowing up memory, and the
    post-conversion check still catches anything that slips through.
    """
    longest = 0
    for ep in episodes:
        path = ep / episode_rel
        if not path.exists():
            continue
        try:
            with h5py.File(path, "r") as h5file:
                longest = max(longest, _episode_frame_count(h5file, mappings))
        except OSError:
            continue

    if longest == 0:
        logging.warning("Could not read any episode length; using encoder queue of 1024 frames.")
        return 1024

    size = min(longest, cap)
    if longest > cap:
        logging.warning(
            f"Longest episode is {longest} frames but the encoder queue is capped at {cap};"
            " frames may still be dropped. The post-conversion check will report it."
        )
    else:
        logging.info(f"Encoder queue: {size} frames (longest episode: {longest}).")
    return size


def _count_frames(mp4: Path, *, exact: bool) -> int:
    """Frame count of one mp4. ``exact`` decodes every packet instead of trusting the header.

    Decoding is accurate but slow (minutes for a multi-GB AV1 file), while the container's
    own count is instant. They agree in practice, so the header is used first and decoding
    only settles a disagreement.
    """
    entry = "stream=nb_read_frames" if exact else "stream=nb_frames"
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if exact:
        cmd.append("-count_frames")
    cmd += ["-show_entries", entry, "-of", "default=nw=1:nk=1", str(mp4)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else -1


def verify_video_frames(dataset_root: Path) -> bool:
    """Check that every camera's video holds exactly as many frames as the parquet data.

    LeRobot's streaming encoder silently drops frames when its queue is full, so a
    conversion can look successful while a camera is short a few frames. Counting them
    back out is the only reliable way to notice.
    """
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        logging.warning(f"verify: info.json not found at {info_path}, skipping check")
        return True

    with open(info_path) as f:
        expected = json.load(f).get("total_frames")
    if not expected:
        logging.warning("verify: total_frames missing from info.json, skipping check")
        return True

    ok = True
    for cam_dir in sorted((dataset_root / "videos").glob("*")):
        if not cam_dir.is_dir():
            continue
        mp4s = sorted(cam_dir.rglob("*.mp4"))
        counts = [_count_frames(m, exact=False) for m in mp4s]
        # Fall back to decoding for any file whose header does not carry a count.
        for idx, (m, n) in enumerate(zip(mp4s, counts, strict=True)):
            if n < 0:
                counts[idx] = _count_frames(m, exact=True)
        total = sum(max(n, 0) for n in counts)

        if total != expected:
            # Only pay for a full decode when the headers disagree, to avoid false alarms.
            total = sum(max(_count_frames(m, exact=True), 0) for m in mp4s)

        if total == expected:
            logging.info(f"verify: {cam_dir.name} {total}/{expected} frames OK")
        else:
            ok = False
            logging.error(f"verify: {cam_dir.name} {total}/{expected} frames — {expected - total} MISSING")
    return ok


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
        "--no-streaming-encoding",
        dest="streaming_encoding",
        action="store_false",
        help=(
            "Fall back to LeRobot's default path, which buffers every frame as a temporary"
            " PNG before encoding. Much slower and writes far more scratch data; frames and"
            " video are unchanged, image statistics come from a sampled subset."
        ),
    )
    parser.set_defaults(streaming_encoding=True)
    parser.add_argument(
        "--encoder-queue-maxsize",
        type=int,
        default=0,
        help=(
            "Frames buffered per camera when streaming to the encoder. LeRobot drops frames"
            " instead of blocking once this queue is full, so it must be able to hold a whole"
            " episode. Default 0 sizes it from the longest episode; each slot costs about"
            " H*W*3 bytes per camera."
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

    # SVT-AV1 encoder log level: 0=silent, 1=error, 2=warn, 3=info(default)
    # Override with SVT_LOG=3 if you need full encoder diagnostics.
    os.environ.setdefault("SVT_LOG", "1")
    # Suppress ffmpeg/libav "moov atom" info logs
    os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")
    logging.getLogger("libav").setLevel(logging.ERROR)

    config = load_config(args.config)

    episode_rel = config.get("episode_path", "data/trajectory.hdf5")
    src_root = Path(args.src_root)
    episodes = sorted([ep for ep in src_root.iterdir() if ep.is_dir()])

    queue_maxsize = args.encoder_queue_maxsize
    if args.streaming_encoding and queue_maxsize <= 0:
        queue_maxsize = plan_encoder_queue(episodes, episode_rel, config["mappings"])

    dataset = initialize_dataset(
        repo_id=args.repo_id,
        tgt_path=args.tgt_path,
        config=config,
        streaming_encoding=args.streaming_encoding,
        encoder_queue_maxsize=queue_maxsize,
    )

    total = len(episodes)
    logging.info(f"Start processing {total} episodes (decode workers per episode: {decode_workers})...")
    saved = 0
    for idx, ep_dir in enumerate(episodes, 1):
        ep_path = ep_dir / episode_rel
        if not ep_path.exists():
            flat_path = ep_dir / "trajectory.hdf5"
            if flat_path.exists():
                ep_path = flat_path
            else:
                logging.warning(f"Episode HDF5 not found: tried {ep_path} and {flat_path}")
        if process_episode(
            ep_path,
            dataset,
            args.task_name,
            config["mappings"],
            decode_workers=decode_workers,
        ):
            dataset.save_episode()
            saved += 1
            logging.info(f"[{saved}/{total}] Saved episode: {ep_dir.name}")

    dataset.finalize()

    if args.stats_override:
        if "stats_override" in config:
            dataset_root = Path(args.tgt_path) / args.repo_id
            apply_stats_override(dataset_root, config["stats_override"])
        else:
            logging.warning("--stats-override flag set but no 'stats_override' section found in config")

    logging.info("Dataset conversion completed!")

    if not verify_video_frames(Path(args.tgt_path) / args.repo_id):
        logging.error(
            "Frame count mismatch: the encoder dropped frames. Re-run with a larger"
            " --encoder-queue-maxsize, or with --no-streaming-encoding."
        )
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
