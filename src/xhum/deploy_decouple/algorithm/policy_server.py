#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remote policy server (Python 3.12 + LeRobot).

ROS / Python 3.10 nodes talk to this process over ZeroMQ REQ/REP — no torch
inside the ROS interpreter.

Protocol (multipart ZMQ):
  Request:  [meta_json_utf8, img_bytes_0, img_bytes_1, ...]
    meta JSON: {
      "version": 1,
      "state": [float, ...],
      "images": {"<short_cam_name>": [H, W, 3], ...}   # order matches following frames
    }
    Each image: row-major uint8 RGB, shape as in meta.

  Reply: [meta_json_utf8, action_float32_bytes]
    meta: {"shape": [1, action_dim]} or error: {"error": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zmq

# Same directory as this file
_DEPLOY_ROOT = Path(__file__).resolve().parent
if str(_DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_ROOT))

from policy_agent import PolicyAgent


def _decode_obs(parts: list[bytes]) -> dict:
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


def _encode_action(tensor) -> tuple[bytes, bytes]:
    arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    meta = json.dumps({"version": 1, "shape": list(arr.shape)})
    return meta.encode("utf-8"), arr.tobytes()


def main():
    parser = argparse.ArgumentParser(description="ZMQ policy server (Py3.12 + LeRobot)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained_model directory")
    parser.add_argument("--bind", type=str, default="tcp://0.0.0.0:5555", help="ZMQ bind address")
    parser.add_argument("--linger", type=int, default=0, help="ZMQ socket linger (ms)")
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, args.linger)
    sock.bind(args.bind)
    print(f"[policy_server] bound {args.bind}", flush=True)

    agent = PolicyAgent(args.model_path)
    print("[policy_server] ready for requests", flush=True)

    while True:
        try:
            parts = sock.recv_multipart()
            obs = _decode_obs(parts)
            out = agent.inference(obs)
            m, b = _encode_action(out)
            sock.send_multipart([m, b])
        except Exception as e:
            err = json.dumps({"version": 1, "error": str(e), "shape": []})
            try:
                sock.send_multipart([err.encode("utf-8"), b""])
            except zmq.ZMQError:
                pass
            print(f"[policy_server] error: {e}", flush=True)


if __name__ == "__main__":
    main()
