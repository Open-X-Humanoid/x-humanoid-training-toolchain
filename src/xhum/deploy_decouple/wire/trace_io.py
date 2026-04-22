#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug-dump helpers shared by the wire's two endpoints.

Called from both ``policy_client`` (pre-send, ROS env, Py 3.10) and
``policy_server`` (post-decode, LeRobot env, Py 3.12). Keep this file's
runtime deps to numpy + stdlib — the optional PNG encoders are imported
inside the writer so a missing dep degrades to a ``.npy`` fallback instead
of crashing one end of the wire.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


def write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    """Write uint8 RGB (H, W, 3) to PNG.

    Tries OpenCV first (faster, usually already installed alongside ROS /
    lerobot), then Pillow, then falls back to ``.npy``. We never raise if
    encoding fails — the caller has already incremented a counter and
    expects at most a log line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    try:
        import cv2

        # OpenCV stores as BGR; our wire/in-memory format is RGB.
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise OSError("cv2.imwrite returned False")
        return
    except Exception:
        pass
    try:
        from PIL import Image

        Image.fromarray(arr, mode="RGB").save(path)
        return
    except Exception:
        pass
    # Last resort: raw numpy dump under a sibling .npy path so nothing is lost.
    np.save(path.with_suffix(".npy"), arr)


def safe_cam_token(name: str) -> str:
    """Filesystem-safe token from a camera short name (for PNG filenames)."""
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return s[:120] if s else "cam"


def coerce_nonneg_int(val: Any, default: int) -> int:
    """YAML may yield bool (yes/no); ``int(True)==1`` would wrongly cap ``max_frames`` to one step."""
    if val is None:
        return default
    if isinstance(val, bool):
        return default
    return int(val)


def maybe_save_obs_rgb_pngs(
    cfg: dict[str, Any] | None,
    obs: Mapping[str, Any],
    seq: int,
    *,
    log_error: Callable[[str], None] | None = None,
) -> None:
    """Write one PNG per camera in ``obs['images']`` when rate limits allow.

    ``cfg`` keys:
      ``dir``        : ``Path`` — output directory (already created)
      ``interval``   : ``int >= 1`` — save every Nth call
      ``max_frames`` : ``int`` — stop after this many *successful* steps (0 = no cap)
      ``saved``      : ``int`` — runtime counter; incremented when **at least one**
                       camera PNG was written for this step (multi-camera sessions
                       still count as one "saved step", so max_frames is step-based
                       not file-based)
    """
    if not cfg:
        return
    if cfg["max_frames"] and cfg["saved"] >= cfg["max_frames"]:
        return
    if seq % cfg["interval"] != 0:
        return
    images = obs.get("images")
    if not isinstance(images, dict) or not images:
        return

    err = log_error if log_error is not None else (lambda m: print(m, flush=True))
    out_dir: Path = cfg["dir"]
    wrote = 0
    for cam_name, img in images.items():
        token = safe_cam_token(str(cam_name))
        path = out_dir / f"{token}_{seq:08d}.png"
        try:
            write_rgb_png(path, img)
            wrote += 1
        except Exception as e:
            err(f"image_save: failed to write {path}: {e}")
    if wrote:
        cfg["saved"] += 1


def save_joints_vector(
    root: Path,
    seq: int,
    joints: np.ndarray,
    *,
    log_error: Callable[[str], None] | None = None,
) -> None:
    """Save ``arm_gripper_joints`` as float32 (N,) under ``root/state_{seq:08d}.npy``."""
    err = log_error if log_error is not None else (lambda m: print(m, flush=True))
    root.mkdir(parents=True, exist_ok=True)
    vec = np.asarray(joints, dtype=np.float32).reshape(-1)
    path = root / f"state_{seq:08d}.npy"
    try:
        np.save(path, vec)
    except Exception as e:
        err(f"joints: failed to write {path}: {e}")

