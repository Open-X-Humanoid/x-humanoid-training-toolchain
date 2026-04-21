#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared ZMQ observation multipart encode/decode (``policy_client`` ↔ ``policy_server``)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def obs_to_multipart(obs: dict[str, Any]) -> list[bytes]:
    """Build REQ multipart: ``[meta_json, img0_bytes, ...]`` (same wire format as PolicyClient)."""
    images = obs["images"]
    state = np.asarray(obs["arm_gripper_joints"], dtype=np.float32)
    meta: dict[str, Any] = {
        "version": 1,
        "state": state.reshape(-1).tolist(),
        "images": {name: list(img.shape) for name, img in images.items()},
    }
    parts: list[bytes] = [json.dumps(meta).encode("utf-8")]
    for name in meta["images"]:
        img = images[name]
        if img.dtype != np.uint8:
            img = np.asarray(img, dtype=np.uint8)
        parts.append(img.tobytes(order="C"))
    return parts


def multipart_to_obs(parts: list[bytes]) -> dict[str, Any]:
    """Decode REQ multipart into observation dict (same rules as policy server)."""
    if len(parts) < 1:
        raise ValueError("empty message")
    meta = json.loads(parts[0].decode("utf-8"))
    if meta.get("version") != 1:
        raise ValueError(f"unsupported protocol version: {meta.get('version')}")

    state = np.asarray(meta["state"], dtype=np.float32)
    images_meta = meta["images"]
    if not isinstance(images_meta, dict):
        raise ValueError("meta.images must be a dict")

    obs_images: dict[str, np.ndarray] = {}
    idx = 1
    for cam_name, shape in images_meta.items():
        if idx >= len(parts):
            raise ValueError(f"missing image frame for camera {cam_name}")
        h, w, c = int(shape[0]), int(shape[1]), int(shape[2])
        buf = parts[idx]
        idx += 1
        arr = np.frombuffer(memoryview(buf), dtype=np.uint8)
        if arr.size != h * w * c:
            raise ValueError(f"image size mismatch for {cam_name}: expected {h*w*c}, got {arr.size}")
        obs_images[cam_name] = arr.reshape(h, w, c)

    return {"images": obs_images, "arm_gripper_joints": state}

