#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect HDF5 episode files — list all keys, shapes, and sample values.

Run this BEFORE editing your conversion config, so you know exactly what
keys are available and how to assemble them into features.

Usage:
    # Inspect a single HDF5 file
    python inspect_h5.py /path/to/trajectory.hdf5

    # Inspect the first episode under an src_root (auto-finds trajectory.hdf5)
    python inspect_h5.py /path/to/success_episodes

    # Specify a custom episode_path pattern
    python inspect_h5.py /path/to/success_episodes --episode_path data/trajectory.hdf5

    # Show more sample rows
    python inspect_h5.py /path/to/trajectory.hdf5 --rows 5

    # Generate a config template with all numeric keys pre-filled
    python inspect_h5.py /path/to/trajectory.hdf5 --gen_config output_config.json
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def collect_datasets(h5file: h5py.File) -> list[dict]:
    """Walk HDF5 file and collect metadata for every dataset."""
    datasets = []

    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            info = {
                "key": name,
                "shape": obj.shape,
                "dtype": str(obj.dtype),
                "is_numeric": obj.dtype.kind in ("f", "i", "u"),
                "is_object": obj.dtype == object,
            }
            if info["is_numeric"] and len(obj.shape) >= 2:
                info["dim"] = obj.shape[1]
            elif info["is_numeric"] and len(obj.shape) == 1:
                info["dim"] = 1
            else:
                info["dim"] = None
            datasets.append(info)

    h5file.visititems(_visitor)
    return datasets


def print_datasets(h5file: h5py.File, datasets: list[dict], sample_rows: int = 2):
    """Pretty-print all datasets in the HDF5 file."""
    print(f"\n{'=' * 72}")
    print(f"  HDF5 Keys Summary  ({len(datasets)} datasets)")
    print(f"{'=' * 72}\n")

    numeric_ds = [d for d in datasets if d["is_numeric"]]
    image_ds = [d for d in datasets if d["is_object"]]
    other_ds = [d for d in datasets if not d["is_numeric"] and not d["is_object"]]

    if numeric_ds:
        print("── Numeric datasets (candidates for state / action) ──\n")
        print(f"  {'Key':<50} {'Shape':<18} {'Dtype':<10} {'Dim'}")
        print(f"  {'─' * 50} {'─' * 18} {'─' * 10} {'─' * 5}")
        for d in numeric_ds:
            print(f"  {d['key']:<50} {str(d['shape']):<18} {d['dtype']:<10} {d['dim']}")
        print()

        for d in numeric_ds:
            if d["dim"] is not None and d["dim"] <= 50:
                data = np.array(h5file[d["key"]])
                rows_to_show = min(sample_rows, data.shape[0])
                print(f"  >> {d['key']}  (showing first {rows_to_show} rows)")
                for r in range(rows_to_show):
                    vals = data[r]
                    if hasattr(vals, "__len__"):
                        formatted = [f"{v:.4f}" for v in vals]
                        idx_labels = "  ".join(
                            f"[{i}]{v}" for i, v in enumerate(formatted)
                        )
                        print(f"     row {r}: {idx_labels}")
                    else:
                        print(f"     row {r}: {vals:.4f}")
                print()

    if image_ds:
        print("── Image / compressed datasets ──\n")
        print(f"  {'Key':<50} {'Frames':<10} {'Dtype'}")
        print(f"  {'─' * 50} {'─' * 10} {'─' * 10}")
        for d in image_ds:
            print(f"  {d['key']:<50} {d['shape'][0]:<10} {d['dtype']}")
        print()

    if other_ds:
        print("── Other datasets ──\n")
        for d in other_ds:
            print(f"  {d['key']}: shape={d['shape']} dtype={d['dtype']}")
        print()


def print_concat_planner(datasets: list[dict]):
    """Print a concat planning helper."""
    numeric_ds = [d for d in datasets if d["is_numeric"] and d["dim"] is not None]
    if not numeric_ds:
        return

    print(f"{'=' * 72}")
    print("  Concat Planner")
    print(f"{'=' * 72}\n")
    print("  Below are all numeric keys with their dimensions.")
    print("  Pick keys to concatenate for your action / state features.\n")

    total_table = []
    for i, d in enumerate(numeric_ds):
        total_table.append((i, d["key"], d["dim"]))
        print(f"  [{i}]  {d['key']:<50} dim={d['dim']}")

    print()
    print("  Example concat combinations:")
    print()

    if len(numeric_ds) >= 2:
        k0, k1 = numeric_ds[0], numeric_ds[1]
        combined_dim = (k0["dim"] or 0) + (k1["dim"] or 0)
        print(f"    \"{k0['key']}\" + \"{k1['key']}\"  →  dim={combined_dim}")
        print()
        print("    Config mapping (hdf5_keys concat):")
        print(f'    {{')
        print(f'        "hdf5_keys": ["{k0["key"]}", "{k1["key"]}"],')
        print(f'        "feature_key": "action"')
        print(f'    }}')

    print()
    print("    Config mapping (single key):")
    if numeric_ds:
        k = numeric_ds[0]
        print(f'    {{')
        print(f'        "hdf5_key": "{k["key"]}",')
        print(f'        "feature_key": "observation.state"')
        print(f'    }}')

    print()
    print("    Config mapping (single key with column slice):")
    if numeric_ds:
        k = numeric_ds[0]
        print(f'    {{')
        print(f'        "hdf5_key": "{k["key"]}",')
        print(f'        "slice": [0, 7],')
        print(f'        "feature_key": "action"')
        print(f'    }}')
    print()


def generate_config_template(datasets: list[dict], output_path: str):
    """Generate a starter config JSON with all numeric keys listed."""
    numeric_ds = [d for d in datasets if d["is_numeric"] and d["dim"] is not None]
    image_ds = [d for d in datasets if d["is_object"]]

    features = {}
    mappings = []

    for d in numeric_ds:
        feature_key = d["key"].replace("/", ".")
        features[feature_key] = {
            "dtype": "float32",
            "shape": [d["dim"]],
            "names": None,
        }
        mappings.append({
            "hdf5_key": d["key"],
            "feature_key": feature_key,
            "_comment": f"dim={d['dim']}",
        })

    for d in image_ds:
        feature_key = d["key"].replace("/", ".").replace("observations.", "observation.")
        features[feature_key] = {
            "dtype": "video",
            "shape": [360, 640, 3],
            "names": ["height", "width", "channels"],
        }
        mappings.append({
            "hdf5_key": d["key"],
            "feature_key": feature_key,
            "decode": "jpeg",
            "resize": [640, 360],
        })

    config = {
        "_README": "Auto-generated template. Edit features/mappings as needed.",
        "dataset": {"fps": 30, "robot_type": "tienkung"},
        "episode_path": "data/trajectory.hdf5",
        "features": features,
        "mappings": mappings,
        "stats_override": {
            "_comment": "Add manual stats here after deciding your feature layout, e.g.:",
            "_example": {
                "action": {
                    "mean": ["...fill 26 values..."],
                    "std": ["...fill 26 values..."],
                }
            },
        },
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=4, default=str)

    print(f"\nConfig template written to: {output_path}")
    print("Edit the features, mappings, and stats_override sections as needed.\n")


def find_h5_file(path: Path, episode_path: str = "data/trajectory.hdf5") -> Path:
    """Resolve a path to a concrete HDF5 file."""
    if path.is_file() and (path.suffix in (".hdf5", ".h5")):
        return path

    if path.is_dir():
        direct = path / episode_path
        if direct.exists():
            return direct

        for sub in sorted(path.iterdir()):
            if sub.is_dir():
                candidate = sub / episode_path
                if candidate.exists():
                    print(f"Using first episode found: {candidate}")
                    return candidate

    print(f"ERROR: Could not find HDF5 file at {path}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect HDF5 episode files — discover keys before conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to HDF5 file, or src_root directory containing episodes",
    )
    parser.add_argument(
        "--episode_path",
        type=str,
        default="data/trajectory.hdf5",
        help="Relative path to HDF5 inside each episode dir (default: data/trajectory.hdf5)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=2,
        help="Number of sample rows to display (default: 2)",
    )
    parser.add_argument(
        "--gen_config",
        type=str,
        default=None,
        help="Generate a config template JSON at this path",
    )
    args = parser.parse_args()

    h5_path = find_h5_file(Path(args.path), args.episode_path)
    print(f"Inspecting: {h5_path}")

    with h5py.File(h5_path, "r") as h5file:
        datasets = collect_datasets(h5file)
        print_datasets(h5file, datasets, sample_rows=args.rows)
        print_concat_planner(datasets)

    if args.gen_config:
        generate_config_template(datasets, args.gen_config)


if __name__ == "__main__":
    main()
