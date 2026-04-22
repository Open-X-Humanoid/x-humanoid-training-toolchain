# -*- coding: utf-8 -*-
"""Load observation trajectories from HDF5 for replay-over-ZMQ (no ROS imports).

Public API:
  - ``load_replay_obs_trajectory`` — used by ``mode=replay`` / ``mode=replay_debug``
  - ``load_aligned_joint_timeseries`` — kept for external offline eval scripts
    (e.g. ``scripts/eval_policy_from_hdf5.py``); not called in-tree
"""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from typing import Any, Tuple

import h5py
import numpy as np


def _to_uint8_hwc(frame: np.ndarray) -> np.ndarray:
    """Return (H, W, 3) uint8 RGB."""
    x = np.asarray(frame)
    if x.ndim != 3:
        raise ValueError(f"image frame must be 3d, got shape {x.shape}")

    if x.dtype != np.uint8:
        xf = x.astype(np.float32)
        if xf.size and xf.max() <= 1.0 + 1e-5:
            xf = xf * 255.0
        x = np.clip(xf, 0, 255).astype(np.uint8)

    if x.shape[-1] == 3:
        return x
    if x.shape[0] == 3:
        return np.transpose(x, (1, 2, 0))
    raise ValueError(f"cannot infer HWC layout from shape {x.shape}")


def _decode_jpeg_cell(cell: Any) -> np.ndarray:
    """Decode one HDF5 object cell (uint8 1d JPEG bytes) to RGB uint8 (H, W, 3)."""
    if isinstance(cell, np.ndarray) and cell.dtype == np.uint8 and cell.ndim == 1:
        buf = cell.tobytes()
    elif isinstance(cell, (bytes, bytearray)):
        buf = bytes(cell)
    else:
        buf = np.asarray(cell, dtype=np.uint8).tobytes()

    try:
        import cv2

        arr = np.frombuffer(buf, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    from PIL import Image

    im = Image.open(BytesIO(buf)).convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def _dataset_is_jpeg_object_stack(ds: h5py.Dataset) -> bool:
    if not isinstance(ds, h5py.Dataset) or ds.dtype != object or ds.ndim != 1 or ds.shape[0] < 1:
        return False
    try:
        x0 = ds[0]
        if isinstance(x0, np.ndarray) and x0.dtype == np.uint8 and x0.ndim == 1 and x0.size > 100:
            _decode_jpeg_cell(x0)
            return True
    except Exception:
        return False
    return False


def _discover_rgb_video_dataset(f: h5py.File) -> h5py.Dataset | None:
    """Last resort: any 4D dataset that looks like (T,H,W,3) or (T,3,H,W)."""
    best: h5py.Dataset | None = None
    best_rank = -1

    def visitor(name: str, obj: Any) -> None:
        nonlocal best, best_rank
        if not isinstance(obj, h5py.Dataset) or obj.ndim != 4:
            return
        s = obj.shape
        ok = False
        if s[-1] == 3 and min(s[1], s[2]) >= 4:
            ok = True
        elif s[1] == 3 and min(s[2], s[3]) >= 4:
            ok = True
        if not ok:
            return
        n = name.lower()
        rank = 0
        if "image" in n or "rgb" in n or "color" in n:
            rank += 4
        if "camera" in n or "cam" in n:
            rank += 2
        if "depth" in n and "rgb" not in n:
            rank -= 3
        if rank > best_rank:
            best_rank = rank
            best = obj

    f.visititems(visitor)
    return best


def _discover_jpeg_object_stack(f: h5py.File) -> h5py.Dataset | None:
    """Find a (T,) object dataset of JPEG byte rows (RoboMIND-style rgb_images)."""
    best: h5py.Dataset | None = None
    best_rank = -1

    def visitor(name: str, obj: Any) -> None:
        nonlocal best, best_rank
        if not isinstance(obj, h5py.Dataset) or not _dataset_is_jpeg_object_stack(obj):
            return
        n = name.lower()
        rank = 0
        if "rgb" in n:
            rank += 6
        if "image" in n:
            rank += 2
        if "depth" in n and "rgb" not in n:
            rank -= 8
        if rank > best_rank:
            best_rank = rank
            best = obj

    f.visititems(visitor)
    return best


def _rgb_images_group_pick(grp: h5py.Group, obs_camera_key: str) -> h5py.Dataset:
    keys = sorted(grp.keys())
    if not keys:
        raise KeyError("empty rgb_images group")
    if obs_camera_key in keys:
        return grp[obs_camera_key]
    if len(keys) == 1:
        return grp[keys[0]]
    if obs_camera_key == "camera" and "camera_camera" in keys:
        return grp["camera_camera"]
    if obs_camera_key == "camera_head" and "camera_head_camera" in keys:
        return grp["camera_head_camera"]
    for k in keys:
        if obs_camera_key in k or k.startswith(obs_camera_key):
            return grp[k]
    raise KeyError(
        f"observations/rgb_images has keys {keys}; set replay_images_h5_key "
        f"(model / obs uses {obs_camera_key!r})"
    )


def _resolve_image_dataset(
    f: h5py.File,
    obs_camera_key: str,
    images_h5_key: str | None,
) -> h5py.Dataset:
    if images_h5_key:
        if images_h5_key not in f:
            hint = ""
            if "observations/rgb_images/camera_camera" in f:
                hint = (
                    " This HDF5 stores JPEG frames at "
                    "'observations/rgb_images/camera_camera' (not observations/images/...)."
                )
            raise KeyError(f"replay_images_h5_key not found: {images_h5_key!r}.{hint}")
        return f[images_h5_key]

    candidates = [
        f"observations/images/{obs_camera_key}",
        f"observation/images/{obs_camera_key}",
        "observations/images/camera_head",
        "observation/images/camera_head",
        "observations/images/camera",
        "observation/images/camera",
        "observations/rgb_images/camera_camera",
        "observation/rgb_images/camera_camera",
        f"observations/rgb_images/{obs_camera_key}",
        f"observation/rgb_images/{obs_camera_key}",
    ]
    for path in candidates:
        if path in f:
            return f[path]

    grp = f.get("observations/images") or f.get("observation/images")
    if grp is not None and isinstance(grp, h5py.Group):
        keys = sorted(grp.keys())
        if obs_camera_key in keys:
            return grp[obs_camera_key]
        if len(keys) == 1:
            return grp[keys[0]]
        if obs_camera_key == "camera" and "camera_head" in keys:
            return grp["camera_head"]
        if obs_camera_key == "camera_head" and "camera" in keys:
            return grp["camera"]
        if obs_camera_key not in keys:
            ds = _discover_rgb_video_dataset(f)
            if ds is not None:
                return ds
        raise KeyError(
            f"observations/images has multiple keys {keys}; set replay_images_h5_key in config "
            f"(model asks for {obs_camera_key!r})"
        )

    grp_rgb = f.get("observations/rgb_images") or f.get("observation/rgb_images")
    if grp_rgb is not None and isinstance(grp_rgb, h5py.Group):
        return _rgb_images_group_pick(grp_rgb, obs_camera_key)

    ds = _discover_rgb_video_dataset(f)
    if ds is not None:
        return ds
    ds2 = _discover_jpeg_object_stack(f)
    if ds2 is not None:
        return ds2

    raise KeyError(
        "no RGB dataset found; set replay_images_h5_key. "
        "Many RoboMIND trajectories use JPEG rows at observations/rgb_images/camera_camera."
    )


def _resolve_state_array(f: h5py.File, state_h5_key: str | None) -> np.ndarray:
    if state_h5_key:
        if state_h5_key not in f:
            raise KeyError(f"replay_state_h5_key not found: {state_h5_key!r}")
        return np.asarray(f[state_h5_key])

    if "puppet/joint_position" in f:
        return np.asarray(f["puppet/joint_position"])

    for path in ("observations/state", "observation.state", "observations/qpos", "observation/qpos"):
        if path in f:
            return np.asarray(f[path])

    raise KeyError(
        "no state dataset found; set replay_state_h5_key or use HDF5 with puppet/joint_position"
    )


def _aligned_len(t_img: int, t_st: int, logger: Any) -> int:
    """Return ``min(t_img, t_st)``; warn when the two don't match."""
    t = min(t_img, t_st)
    if t_img != t_st:
        logger.warning(
            f"HDF5 image length {t_img} != state length {t_st}; using first {t} frames"
        )
    return t


def _frame_decoder(
    img_ds: h5py.Dataset | np.ndarray,
) -> Tuple[int, Callable[[int], np.ndarray], str]:
    """Return ``(t_img, decode(i) -> (H,W,3) uint8 RGB, layout_tag)`` for either layout.

    Handles the two HDF5 image shapes we see in the wild — JPEG bytes per row
    (``(T,)`` object) and dense 4D (``(T,H,W,3)`` / ``(T,3,H,W)``) — behind a
    single closure so callers can drop their branching.
    """
    if isinstance(img_ds, h5py.Dataset) and _dataset_is_jpeg_object_stack(img_ds):
        return (
            img_ds.shape[0],
            lambda i: _to_uint8_hwc(_decode_jpeg_cell(img_ds[i])),
            "JPEG stack",
        )
    rgb = np.asarray(img_ds)
    if rgb.ndim != 4:
        raise ValueError(f"image dataset must be 4d (T,H,W,C) or (T,C,H,W), got {rgb.shape}")
    return rgb.shape[0], lambda i: _to_uint8_hwc(rgb[i]), "dense 4D"


def load_replay_obs_trajectory(
    h5_path: str,
    *,
    obs_camera_key: str,
    images_h5_key: str | None,
    state_h5_key: str | None,
    logger: Any,
) -> list[dict[str, Any]]:
    """Return one observation dict per timestep (wire-format for ``PolicyClient``)."""
    with h5py.File(h5_path, "r") as f:
        img_ds = _resolve_image_dataset(f, obs_camera_key, images_h5_key)
        state_arr = np.asarray(_resolve_state_array(f, state_h5_key), dtype=np.float32)
        if state_arr.ndim != 2:
            raise ValueError(f"state dataset must be 2d (T, D), got {state_arr.shape}")

        t_img, decode, layout = _frame_decoder(img_ds)
        t = _aligned_len(t_img, state_arr.shape[0], logger)

        out: list[dict[str, Any]] = [
            {
                "images": {obs_camera_key: decode(i)},
                "arm_gripper_joints": state_arr[i].reshape(-1),
            }
            for i in range(t)
        ]
        logger.info(
            f"Loaded {len(out)} replay observations from {h5_path} "
            f"(camera={obs_camera_key!r}, layout={layout})"
        )
        return out


def load_aligned_joint_timeseries(
    h5_path: str,
    *,
    obs_camera_key: str,
    images_h5_key: str | None,
    state_h5_key: str | None,
    logger: Any,
) -> np.ndarray:
    """Aligned joint/state array ``(T, D)`` — same ``T`` as ``load_replay_obs_trajectory``.

    Public API for **external** offline eval scripts (``scripts/eval_policy_from_hdf5.py``
    and friends, which are gitignored). Not called in-tree — kept here so those
    scripts don't duplicate the image/state alignment logic.
    """
    with h5py.File(h5_path, "r") as f:
        img_ds = _resolve_image_dataset(f, obs_camera_key, images_h5_key)
        state_arr = np.asarray(_resolve_state_array(f, state_h5_key), dtype=np.float32)
        if state_arr.ndim != 2:
            raise ValueError(f"state dataset must be 2d (T, D), got {state_arr.shape}")

        t_img, _decode, _layout = _frame_decoder(img_ds)
        t = _aligned_len(t_img, state_arr.shape[0], logger)
        return state_arr[:t]
