#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZMQ client for remote policy inference (Python 3.10; numpy + pyzmq only).

Observation dict:
  images: short camera name -> uint8 RGB (H, W, 3)
  arm_gripper_joints: 1d state vector

Returns action ndarray shape (1, action_dim) float32 on CPU.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import zmq


class PolicyClient:
    def __init__(self, server_url: str = "tcp://127.0.0.1:5555", timeout_ms: int = 10_000):
        self._server_url = server_url
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._sock.connect(server_url)

    def close(self):
        self._sock.close(linger=0)
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *args: Any):
        self.close()

    def reset(self):
        pass

    def inference(self, obs: dict) -> np.ndarray:
        images = obs["images"]
        state = np.asarray(obs["arm_gripper_joints"], dtype=np.float32)

        meta: dict = {
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

        self._sock.send_multipart(parts)
        reply = self._sock.recv_multipart()
        rmeta = json.loads(reply[0].decode("utf-8"))
        if rmeta.get("error"):
            raise RuntimeError(rmeta["error"])
        shape = rmeta.get("shape") or []
        buf = reply[1] if len(reply) > 1 else b""
        if not shape or not buf:
            raise RuntimeError("invalid reply: %s" % (rmeta,))
        arr = np.frombuffer(memoryview(buf), dtype=np.float32).reshape(shape)
        return arr
