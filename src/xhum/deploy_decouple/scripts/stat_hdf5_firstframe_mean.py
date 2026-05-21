#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Average the first frame of one or more dataset keys across many HDF5 files in a directory.

For each ``*.hdf5`` / ``*.h5`` under ``--dir`` (optionally recursive), reads ``dataset[0]``
from each ``--key`` and computes the element-wise mean over files. For a given key, all
first frames must have the same shape (after ``np.asarray``).

Writes a single ``<stem>.json`` summarizing every key, with the per-key mean stored
inline as a nested list under ``keys_meta[*].mean``.

Requires: Python 3.10+, numpy, h5py.

Example (single key)::

  python scripts/stat_hdf5_firstframe_mean.py \\
    --dir /data/episodes/success \\
    --key puppet/joint_position \\
    --output ./first_mean

Example (multiple keys, one pass over files)::

  python scripts/stat_hdf5_firstframe_mean.py \\
    --dir /data/episodes/success -r \\
    --key puppet/joint_position master/joint_position \\
    --output ./first_mean
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


def _collect_h5_paths(root: Path, recursive: bool) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    patterns = ("*.hdf5", "*.h5")
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in patterns:
        it = root.rglob(pat) if recursive else root.glob(pat)
        for p in it:
            rp = p.resolve()
            if rp.is_file() and rp not in seen:
                seen.add(rp)
                out.append(p)
    return sorted(out, key=lambda x: str(x))


def _read_first_frames(path: Path, keys: list[str]) -> dict[str, np.ndarray | Exception]:
    """Read dataset[key][0] for each key from a single HDF5 file.

    Returns a dict mapping key -> ndarray on success, or key -> Exception on failure.
    The file is opened only once.
    """
    out: dict[str, np.ndarray | Exception] = {}
    with h5py.File(path, "r") as f:
        for key in keys:
            try:
                if key not in f:
                    raise KeyError(f"missing key {key!r}")
                ds = f[key]
                if ds.shape[0] < 1:
                    raise ValueError(f"empty along axis 0: {key} shape={ds.shape}")
                out[key] = np.asarray(ds[0], dtype=np.float64)
            except Exception as e:
                out[key] = e
    return out


def _safe_key(key: str) -> str:
    return key.replace("/", "_").replace("\\", "_")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Mean of dataset[key][0] over all HDF5 files in a folder.",
    )
    p.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory containing trajectory.hdf5 (or similar) files.",
    )
    p.add_argument(
        "--key",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more HDF5 dataset paths, e.g. "
            "--key puppet/joint_position master/joint_position."
        ),
    )
    p.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search recursively for *.hdf5 / *.h5.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path stem (writes STEM.json containing per-key mean values inline). "
            "Default: ./firstframe_mean."
        ),
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any file is skipped (missing key, shape mismatch, etc.).",
    )
    args = p.parse_args()

    root = args.dir.expanduser().resolve()
    paths = _collect_h5_paths(root, args.recursive)
    if not paths:
        print(f"ERROR: no .hdf5/.h5 under {root!r} (recursive={args.recursive})", file=sys.stderr)
        return 1

    keys: list[str] = list(dict.fromkeys(args.key))
    dup = len(args.key) - len(keys)
    if dup:
        print(f"WARN: removed {dup} duplicate --key entries", file=sys.stderr)

    stem = args.output if args.output is not None else Path.cwd() / "firstframe_mean"
    stem = stem.expanduser().resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = stem.with_suffix(".json")

    per_key_vecs: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    per_key_used: dict[str, list[str]] = {k: [] for k in keys}
    per_key_skipped: dict[str, list[dict[str, str]]] = {k: [] for k in keys}
    per_key_ref_shape: dict[str, tuple[int, ...] | None] = {k: None for k in keys}

    for fp in paths:
        try:
            results = _read_first_frames(fp, keys)
        except Exception as e:
            for k in keys:
                per_key_skipped[k].append({"file": str(fp), "reason": f"open failed: {e}"})
            if args.strict:
                print(f"ERROR strict: {fp}: open failed: {e}", file=sys.stderr)
                return 1
            continue

        for k in keys:
            r = results[k]
            if isinstance(r, Exception):
                per_key_skipped[k].append({"file": str(fp), "reason": str(r)})
                if args.strict:
                    print(f"ERROR strict: {fp} [{k}]: {r}", file=sys.stderr)
                    return 1
                continue
            ref = per_key_ref_shape[k]
            if ref is None:
                per_key_ref_shape[k] = r.shape
            elif r.shape != ref:
                msg = f"shape {r.shape} != reference {ref}"
                per_key_skipped[k].append({"file": str(fp), "reason": msg})
                if args.strict:
                    print(f"ERROR strict: {fp} [{k}]: {msg}", file=sys.stderr)
                    return 1
                continue
            per_key_vecs[k].append(r)
            per_key_used[k].append(str(fp))

    keys_meta: list[dict] = []
    any_ok = False
    for k in keys:
        vecs = per_key_vecs[k]
        used = per_key_used[k]
        skipped = per_key_skipped[k]

        if not vecs:
            print(
                f"ERROR[{k}]: no usable files after filtering (all skipped).",
                file=sys.stderr,
            )
            for s in skipped[:10]:
                print(f"  skip: {s['file']}: {s['reason']}", file=sys.stderr)
            if len(skipped) > 10:
                print(f"  ... and {len(skipped) - 10} more", file=sys.stderr)
            keys_meta.append({
                "key": k,
                "ok": False,
                "n_files_used": 0,
                "n_files_skipped": len(skipped),
                "files_used": used,
                "files_skipped": skipped,
            })
            continue

        stacked = np.stack(vecs, axis=0)
        mean_arr = stacked.mean(axis=0).astype(np.float64, copy=False)
        any_ok = True

        print(
            f"OK[{k}]: used {len(used)}/{len(paths)} files | shape={mean_arr.shape}",
            flush=True,
        )
        if skipped:
            print(f"WARN[{k}]: skipped {len(skipped)} file(s); see {json_path!s}", flush=True)

        keys_meta.append({
            "key": k,
            "ok": True,
            "shape": list(mean_arr.shape),
            "dtype": "float64",
            "mean": mean_arr.tolist(),
            "n_files_used": len(used),
            "n_files_skipped": len(skipped),
            "files_used": used,
            "files_skipped": skipped,
        })

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "directory": str(root),
        "keys": keys,
        "recursive": args.recursive,
        "n_files_found": len(paths),
        "keys_meta": keys_meta,
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote summary {json_path!s}", flush=True)

    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
