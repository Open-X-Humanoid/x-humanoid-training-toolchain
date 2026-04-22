#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless HDF5 → ZMQ replay loop (no rclpy).

Reads observations from a trajectory file, sends each step to ``policy_server``,
logs the returned action. ``run.py`` dispatches here when the YAML has
``mode: replay_debug`` so ROS / rclpy is never imported on this path.

This is the cheapest way to validate the full wire + model pipeline on a
laptop without sourcing ROS: same ``PolicyClient`` + ``load_config`` + HDF5
loader as ``mode=replay``, just no ROS publishers on the output side.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config_loader import load_config, make_policy_client
from replay_io.hdf5_replay_obs import load_replay_obs_trajectory


class _StdioLogger:
    """Minimal info/warning/error shim used before rclpy is available."""

    def info(self, msg: str) -> None:
        print(msg)

    def warning(self, msg: str) -> None:
        print(f"[WARN] {msg}", file=sys.stderr)

    def error(self, msg: str) -> None:
        print(f"[ERROR] {msg}", file=sys.stderr)


def run(config_path: str) -> int:
    log = _StdioLogger()
    cfg = load_config(config_path, log)
    if cfg.get("mode") != "replay_debug":
        log.error(f"mode must be 'replay_debug' (got {cfg.get('mode')!r})")
        return 1

    h5 = cfg.get("h5_path")
    if not h5 or str(h5) == "PATH_TO_H5":
        log.error("replay_debug: set h5_path in YAML to trajectory.hdf5")
        return 1

    url = cfg.get("policy_server_url", "")
    if not url:
        log.error("replay_debug: set policy_server_url in YAML")
        return 1

    try:
        obs_list = load_replay_obs_trajectory(
            str(h5),
            obs_camera_key=cfg["obs_camera_key"],
            images_h5_key=cfg.get("replay_images_h5_key") or None,
            state_h5_key=cfg.get("replay_state_h5_key") or None,
            logger=log,
        )
    except Exception as e:
        log.error(f"load HDF5 observations failed: {e}")
        return 1

    if not obs_list:
        log.error("empty observation list")
        return 1

    max_steps = int(cfg.get("replay_debug_max_steps", 0) or 0)
    max_steps = len(obs_list) if max_steps <= 0 else min(max_steps, len(obs_list))

    period = 1.0 / max(float(cfg.get("action_rate", 20.0)), 1e-6)
    zmq_to = int(cfg.get("policy_zmq_timeout_ms", 120_000))
    log.info(
        f"replay_debug: {max_steps} ZMQ inference steps  policy_server={url}  "
        f"policy_zmq_timeout_ms={zmq_to} (0 = unlimited)  "
        f"(no rclpy, no publishers; same PolicyClient path as mode=replay)"
    )
    if (cfg.get("image_save") or {}).get("enabled"):
        log.info("image_save.enabled=true")

    client = make_policy_client(cfg, log)
    try:
        client.reset()
        for i in range(max_steps):
            t0 = time.perf_counter()
            action = client.inference(obs_list[i])
            dt = time.perf_counter() - t0
            row = action[0]
            log.info(
                f"step {i + 1}/{max_steps}  wall={dt:.3f}s  action_shape={tuple(action.shape)}  "
                f"|a|_mean={float(np.mean(np.abs(row))):.4f}"
            )
            time.sleep(period)
    finally:
        client.close()

    log.info("replay_debug: finished OK")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Headless HDF5 → ZMQ replay (no ROS)")
    p.add_argument("--config", type=str, required=True)
    raise SystemExit(run(p.parse_args().config))
