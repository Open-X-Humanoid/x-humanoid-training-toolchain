#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline test: load HDF5 actions the same way as replay mode (no ROS).

Usage:
  cd src/xhum/deploy_decouple
  python scripts/test_replay_hdf5.py /path/to/trajectory.hdf5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DEPLOY = Path(__file__).resolve().parents[1]
_ROS_BRIDGE = _DEPLOY / "ros_bridge"
if str(_ROS_BRIDGE) not in sys.path:
    sys.path.insert(0, str(_ROS_BRIDGE))

from hdf5_actions import load_action_trajectory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5_path", type=Path, help="Path to trajectory.hdf5")
    args = ap.parse_args()
    p = args.h5_path.resolve()
    if not p.is_file():
        print(f"ERROR: not a file: {p}", file=sys.stderr)
        return 1
    try:
        actions = load_action_trajectory(str(p))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not actions:
        print("ERROR: zero actions loaded", file=sys.stderr)
        return 1

    a0 = actions[0]
    print(f"OK: loaded {len(actions)} steps, action_dim={a0.shape[0]} dtype={a0.dtype}")
    print(f"  first action min/max: {a0.min():.4f} / {a0.max():.4f}")
    al = actions[-1]
    print(f"  last  action min/max: {al.min():.4f} / {al.max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
