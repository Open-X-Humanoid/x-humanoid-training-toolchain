#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline HDF5 evaluation: PolicyAgent.inference vs next-frame GT joints.

Runs in the Py 3.12 + LeRobot env only (imports ``PolicyAgent``). No ZMQ, no
policy_server — direct in-process inference, so this is the fastest path to
check "does this checkpoint produce sane actions on my recorded data".

For step ``i``:
    pred    = policy_agent.inference(obs[i])[0]     # one action vector
    gt_next = gt_array[i + 1]                       # next-frame target joints
    diff    = pred - gt_next

Why next-frame GT, and why typically ``master/joint_position``:
    ACT learns to imitate the *operator* (master) command stream. At step i the
    model's output is "the command I would issue now", which in teleop data
    maps to the master command recorded at t+1.

All HDF5 dataset paths + the checkpoint's camera short name are **required**
CLI arguments — auto-detection existed earlier but silently picked wrong
datasets on trajectories with multiple candidates. Forcing explicit keys
makes every run auditable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

# sys.path injection: no pip install needed; keeps the env-boundary layout intact.
# PolicyAgent (with its LeRobot deps) is imported lazily inside main() so that
# ``--help`` works in a plain Py 3.12 without LeRobot installed.
_SCRIPTS = Path(__file__).resolve().parent
_DD = _SCRIPTS.parent
for _p in (_DD / "policy", _DD / "robot"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from replay_io.hdf5_replay_obs import load_replay_obs_trajectory  # noqa: E402

log = logging.getLogger("eval")


def _load_gt(h5_path: Path, gt_key: str) -> np.ndarray:
    with h5py.File(str(h5_path), "r") as f:
        if gt_key not in f:
            raise KeyError(f"gt_key not found in HDF5: {gt_key!r}")
        arr = np.asarray(f[gt_key], dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"gt array must be 2d (T, D), got {arr.shape}")
    return arr


def _fmt(vec: np.ndarray, digits: int = 4) -> str:
    with np.printoptions(precision=digits, suppress=True, linewidth=240):
        return str(vec)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline pred-vs-gt eval from HDF5")
    p.add_argument("--h5_path", type=Path, required=True)
    p.add_argument("--model_path", type=Path, required=True,
                   help="Path to the pretrained_model/ directory")
    p.add_argument("--obs_camera_key", type=str, required=True,
                   help="Short camera name the checkpoint expects — must match "
                        "`observation.images.<X>` in the checkpoint's input_features. "
                        "Example: camera_head")
    p.add_argument("--replay_images_h5_key", type=str, required=True,
                   help="HDF5 dataset path for RGB frames. "
                        "Example: observations/rgb_images/camera_camera")
    p.add_argument("--replay_state_h5_key", type=str, required=True,
                   help="HDF5 dataset path for the state vector fed into the model. "
                        "Example: puppet/joint_position")
    p.add_argument("--gt_key", type=str, required=True,
                   help="HDF5 dataset of the ground-truth action stream to compare "
                        "model output against. Example: master/joint_position")
    p.add_argument("--start", type=int, default=0, help="First step index (default: 0)")
    p.add_argument("--max_steps", type=int, default=0,
                   help="Max inference steps (0 = run until trajectory end)")
    p.add_argument("--quiet", action="store_true",
                   help="Skip per-step vector prints; only final per-dim stats")
    return p.parse_args()


def _iter_range(start: int, n_obs: int, n_gt: int, max_steps: int) -> range:
    # ``i + 1`` must index gt, so the last usable i is ``n_gt - 2``.
    t_end = min(n_obs, n_gt - 1)
    if max_steps > 0:
        t_end = min(t_end, start + max_steps)
    if t_end <= start:
        raise ValueError(
            f"nothing to evaluate: start={start}, obs_len={n_obs}, "
            f"gt_len={n_gt} (need start < min(obs_len, gt_len - 1))"
        )
    return range(start, t_end)


def _print_summary(diffs: list[np.ndarray], steps: range) -> None:
    D = np.stack(diffs, axis=0)
    abs_D = np.abs(D)
    print("\n====== summary ======")
    print(f"steps evaluated : {len(diffs)}  (range [{steps.start}, {steps.stop}))")
    print(f"action_dim      : {D.shape[1]}")
    print(f"|diff| per-dim mean : {_fmt(abs_D.mean(axis=0))}")
    print(f"|diff| per-dim max  : {_fmt(abs_D.max(axis=0))}")
    print(f" diff  per-dim min  : {_fmt(D.min(axis=0))}")
    print(f" diff  per-dim max  : {_fmt(D.max(axis=0))}")
    print(f"|diff| overall mean : {float(abs_D.mean()):.6f}")
    print(f"|diff| overall max  : {float(abs_D.max()):.6f}")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from policy_agent import PolicyAgent  # deferred: LeRobot + torch

    obs_list = load_replay_obs_trajectory(
        str(args.h5_path),
        obs_camera_key=args.obs_camera_key,
        images_h5_key=args.replay_images_h5_key,
        state_h5_key=args.replay_state_h5_key,
        logger=log,
    )
    if not obs_list:
        log.error("empty observation list")
        return 1

    gt = _load_gt(args.h5_path, args.gt_key)
    steps = _iter_range(args.start, len(obs_list), gt.shape[0], args.max_steps)

    agent = PolicyAgent(str(args.model_path))
    agent.reset()  # fresh ACT buffer; important because eval may reuse a long-lived agent

    diffs: list[np.ndarray] = []
    for i in steps:
        pred = (
            agent.inference(obs_list[i])
            .detach().cpu().numpy().astype(np.float64).reshape(-1)
        )
        gt_next = gt[i + 1]
        if gt_next.shape != pred.shape:
            log.error(
                f"shape mismatch at step {i}: pred {pred.shape} vs gt_next {gt_next.shape}. "
                "Check --gt_key: is it the action stream (not the raw state)?"
            )
            return 1
        diffs.append(pred - gt_next)

        if not args.quiet:
            print(
                f"--- step {i} (gt_next=row {i + 1}) ---\n"
                f"  pred    = {_fmt(pred)}\n"
                f"  gt_next = {_fmt(gt_next)}\n"
                f"  diff    = {_fmt(diffs[-1])}"
            )

    _print_summary(diffs, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
