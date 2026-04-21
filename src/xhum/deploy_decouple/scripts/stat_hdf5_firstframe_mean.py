#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Average the first frame of a dataset key across many HDF5 files in a directory.

For each ``*.hdf5`` / ``*.h5`` under ``--dir`` (optionally recursive), reads ``dataset[0]``
from ``--key`` and computes the element-wise mean over files. All first frames must have
the same shape (after ``np.asarray``).

Writes ``--output`` as ``.npy`` (mean array) and ``.json`` (metadata).

Requires: Python 3.10+, numpy, h5py.

Example::

  python scripts/stat_hdf5_firstframe_mean.py \\
    --dir /data/episodes/success \\
    --key puppet/joint_position \\
    --output ./puppet_first_mean
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


def _read_first_frame(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"missing key {key!r} in {path}")
        ds = f[key]
        if ds.shape[0] < 1:
            raise ValueError(f"empty along axis 0: {path} {key} shape={ds.shape}")
        return np.asarray(ds[0], dtype=np.float64)


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
        required=True,
        help="HDF5 dataset path, e.g. puppet/joint_position or master/joint_position.",
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
        help="Output path stem (writes STEM.npy and STEM.json). Default: ./firstframe_mean_<key>.",
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

    stem = args.output
    if stem is None:
        safe = args.key.replace("/", "_").replace("\\", "_")
        stem = Path.cwd() / f"firstframe_mean_{safe}"

    stem = stem.expanduser().resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    npy_path = stem.with_suffix(".npy")
    json_path = stem.with_suffix(".json")

    first_vecs: list[np.ndarray] = []
    used: list[str] = []
    skipped: list[dict[str, str]] = []

    ref_shape: tuple[int, ...] | None = None
    for fp in paths:
        try:
            v = _read_first_frame(fp, args.key)
        except Exception as e:
            skipped.append({"file": str(fp), "reason": str(e)})
            if args.strict:
                print(f"ERROR strict: {fp}: {e}", file=sys.stderr)
                return 1
            continue
        if ref_shape is None:
            ref_shape = v.shape
        elif v.shape != ref_shape:
            msg = f"shape {v.shape} != reference {ref_shape}"
            skipped.append({"file": str(fp), "reason": msg})
            if args.strict:
                print(f"ERROR strict: {fp}: {msg}", file=sys.stderr)
                return 1
            continue
        first_vecs.append(v)
        used.append(str(fp))

    if not first_vecs:
        print("ERROR: no usable files after filtering (all skipped).", file=sys.stderr)
        for s in skipped[:20]:
            print(f"  skip: {s['file']}: {s['reason']}", file=sys.stderr)
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more", file=sys.stderr)
        return 1

    stacked = np.stack(first_vecs, axis=0)
    mean_arr = stacked.mean(axis=0)

    np.save(str(npy_path), mean_arr.astype(np.float64, copy=False))

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "directory": str(root),
        "key": args.key,
        "recursive": args.recursive,
        "n_files_found": len(paths),
        "n_files_used": len(used),
        "n_files_skipped": len(skipped),
        "shape": list(mean_arr.shape),
        "dtype": "float64",
        "mean_npy": str(npy_path),
        "files_used": used,
        "files_skipped": skipped,
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"OK: used {len(used)}/{len(paths)} files | shape={mean_arr.shape} | "
        f"wrote {npy_path!s} and {json_path!s}",
        flush=True,
    )
    if skipped:
        print(f"WARN: skipped {len(skipped)} file(s); see {json_path!s}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
