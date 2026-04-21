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
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from utils import coerce_nonneg_int, maybe_save_obs_rgb_pngs, save_joints_vector
from zmq_obs_codec import obs_to_multipart


class PolicyClient:
    def __init__(
        self,
        server_url: str = "tcp://127.0.0.1:5555",
        timeout_ms: int = 120_000,
        *,
        ros_logger: Any = None,
        image_save: dict[str, Any] | None = None,
        joints: dict[str, Any] | None = None,
    ):
        self._server_url = server_url
        self._ctx = zmq.Context()
        to = int(timeout_ms)
        if to <= 0:
            to = -1
        self._zmq_timeout_opt = to
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self._zmq_timeout_opt)
        self._sock.setsockopt(zmq.SNDTIMEO, self._zmq_timeout_opt)
        self._sock.connect(server_url)
        self._ros_logger = ros_logger
        self._inference_seq = 0
        self._img_save: dict[str, Any] | None = None
        self._joints_dump_root: Path | None = None
        jcfg = joints
        if jcfg and bool(jcfg.get("enabled")):
            base = Path(str(jcfg.get("directory", "debug_joints"))).expanduser()
            if not base.is_absolute():
                base = Path.cwd() / base
            if bool(jcfg.get("use_timestamp_subdir", True)):
                base = base / time.strftime("%Y%m%d_%H%M%S")
            self._joints_dump_root = base / "client_pre_send"
            self._joints_dump_root.mkdir(parents=True, exist_ok=True)
            self._log_info(f"joints: saving arm_gripper_joints before ZMQ send under {self._joints_dump_root}")
        if image_save and bool(image_save.get("enabled")):
            base = Path(str(image_save.get("directory", "debug_policy_images"))).expanduser()
            if not base.is_absolute():
                base = Path.cwd() / base
            if bool(image_save.get("use_timestamp_subdir", True)):
                base = base / time.strftime("%Y%m%d_%H%M%S")
            base.mkdir(parents=True, exist_ok=True)
            self._img_save = {
                "dir": base,
                "interval": max(1, coerce_nonneg_int(image_save.get("interval"), 1)),
                "max_frames": max(0, coerce_nonneg_int(image_save.get("max_frames"), 0)),
                "saved": 0,
            }
            self._log_info(
                f"image_save: directory {base} (one PNG per camera after obs encode, before ZMQ send; "
                f"interval={self._img_save['interval']}, max_frames={self._img_save['max_frames']})"
            )

    def close(self):
        self._sock.close(linger=0)
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *args: Any):
        self.close()

    def reset(self):
        pass

    def _recreate_req_socket(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:
            pass
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self._zmq_timeout_opt)
        self._sock.setsockopt(zmq.SNDTIMEO, self._zmq_timeout_opt)
        self._sock.connect(self._server_url)

    @staticmethod
    def _reply_to_action(reply: list[bytes]) -> np.ndarray:
        rmeta = json.loads(reply[0].decode("utf-8"))
        if rmeta.get("error"):
            raise RuntimeError(rmeta["error"])
        shape = rmeta.get("shape") or []
        buf = reply[1] if len(reply) > 1 else b""
        if not shape or not buf:
            raise RuntimeError("invalid reply: %s" % (rmeta,))
        return np.frombuffer(memoryview(buf), dtype=np.float32).reshape(shape)

    def _log_info(self, msg: str) -> None:
        if self._ros_logger is not None:
            self._ros_logger.info(msg)
        else:
            print(msg, flush=True)

    def _log_error(self, msg: str) -> None:
        if self._ros_logger is not None:
            self._ros_logger.error(msg)
        else:
            print(msg, flush=True)

    def _maybe_save_input_images(self, obs: dict) -> None:
        maybe_save_obs_rgb_pngs(
            self._img_save,
            obs,
            self._inference_seq,
            log_error=self._log_error,
        )

    def inference(self, obs: dict) -> np.ndarray:
        self._inference_seq += 1
        parts = obs_to_multipart(obs)
        if self._joints_dump_root is not None:
            save_joints_vector(
                self._joints_dump_root,
                self._inference_seq,
                obs["arm_gripper_joints"],
                log_error=self._log_error,
            )
        self._maybe_save_input_images(obs)

        for attempt in range(2):
            try:
                self._sock.send_multipart(parts)
                reply = self._sock.recv_multipart()
                return self._reply_to_action(reply)
            except zmq.Again:
                if attempt == 0:
                    self._log_error(
                        "ZMQ send/recv timed out (REQ socket must reset). "
                        "Recreating socket and retrying this observation once."
                    )
                    self._recreate_req_socket()
                    continue
                raise RuntimeError(
                    "ZMQ timed out twice or REQ is unusable. "
                    "Increase policy_zmq_timeout_ms in YAML (0 = no limit), "
                    "and ensure policy_server is running."
                ) from None
        raise RuntimeError("ZMQ inference failed")
