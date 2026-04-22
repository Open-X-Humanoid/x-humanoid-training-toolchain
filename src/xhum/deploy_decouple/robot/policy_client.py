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
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq

# Runs in the Py 3.10 ROS env. The wire protocol + trace helpers live one level up
# in ``deploy_decouple/wire`` and are injected into ``sys.path`` at import time so
# this file works without ``pip install``-ing the deploy folder.
_HERE = Path(__file__).resolve().parent
_WIRE = _HERE.parent / "wire"
if str(_WIRE) not in sys.path:
    sys.path.insert(0, str(_WIRE))

from trace_io import coerce_nonneg_int, maybe_save_obs_rgb_pngs, save_joints_vector
from obs_codec import obs_to_multipart, reset_to_multipart


class PolicyClient:
    """Synchronous REQ client. ``timeout_ms=120000`` because first ACT inference
    on CPU can exceed 10s (model warm-up + first batch compile). Pass 0 to
    wait forever — handy for debugging, risky in production."""

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
        # ZMQ convention: -1 = block forever; our public contract uses 0 for that.
        self._zmq_timeout_opt = to if to > 0 else -1
        self._sock = self._new_req_socket()
        self._ros_logger = ros_logger
        self._inference_seq = 0
        self._img_save = _init_image_save_cfg(image_save, self._log_info)
        self._joints_dump_root = _init_joints_dump_root(joints, self._log_info)

    def _new_req_socket(self) -> zmq.Socket:
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._zmq_timeout_opt)
        sock.setsockopt(zmq.SNDTIMEO, self._zmq_timeout_opt)
        sock.connect(self._server_url)
        return sock

    def _recreate_req_socket(self) -> None:
        # REQ sockets enter an unrecoverable "waiting for reply" state after a
        # recv timeout; the only safe recovery is close + new socket. We use
        # linger=0 so the close doesn't block waiting on the server.
        try:
            self._sock.close(linger=0)
        except Exception:
            pass
        self._sock = self._new_req_socket()

    def close(self):
        try:
            self._sock.close(linger=0)
        finally:
            self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *args: Any):
        self.close()

    def reset(self) -> None:
        """Tell the remote policy to clear its per-episode state (ACT action buffer, etc.)."""
        try:
            reply = self._roundtrip(reset_to_multipart())
        except zmq.Again:
            self._recreate_req_socket()
            self._log_error("reset timed out; REQ socket recreated. Remote state may be stale.")
            return
        meta = _decode_reply_meta(reply)
        if meta.get("error"):
            self._log_error(f"remote reset error: {meta['error']}")
        else:
            self._log_info(f"remote reset acknowledged ({meta.get('op', '?')})")

    def inference(self, obs: dict) -> np.ndarray:
        self._inference_seq += 1
        parts = obs_to_multipart(obs)
        # Dump **pre-send**, not on the server, so operators can compare
        # "what the client was about to send" against "what the server decoded".
        # If ZMQ later times out, the disk record of this step is still intact.
        if self._joints_dump_root is not None:
            save_joints_vector(
                self._joints_dump_root,
                self._inference_seq,
                obs["arm_gripper_joints"],
                log_error=self._log_error,
            )
        maybe_save_obs_rgb_pngs(
            self._img_save, obs, self._inference_seq, log_error=self._log_error
        )

        # Retry once on timeout, on a freshly-recreated REQ socket. If the
        # second attempt also times out, we give up — the caller decides
        # whether to abort the episode or move the arm to a safe pose.
        try:
            reply = self._roundtrip(parts)
        except zmq.Again:
            self._log_error(
                "ZMQ send/recv timed out; recreating REQ socket and retrying this observation once."
            )
            self._recreate_req_socket()
            try:
                reply = self._roundtrip(parts)
            except zmq.Again:
                raise RuntimeError(
                    "ZMQ timed out twice; increase policy_zmq_timeout_ms in YAML "
                    "(0 = no limit) and ensure policy_server is running."
                ) from None

        meta = _decode_reply_meta(reply)
        if meta.get("error"):
            raise RuntimeError(meta["error"])
        shape = meta.get("shape") or []
        buf = reply[1] if len(reply) > 1 else b""
        if not shape or not buf:
            raise RuntimeError(f"invalid reply: {meta}")
        return np.frombuffer(memoryview(buf), dtype=np.float32).reshape(shape)

    def _roundtrip(self, parts: list[bytes]) -> list[bytes]:
        self._sock.send_multipart(parts)
        return self._sock.recv_multipart()

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


def _decode_reply_meta(reply: list[bytes]) -> dict[str, Any]:
    if not reply:
        raise RuntimeError("empty reply from policy_server")
    return json.loads(reply[0].decode("utf-8"))


def _init_image_save_cfg(image_save: dict[str, Any] | None, log_info) -> dict[str, Any] | None:
    if not image_save or not bool(image_save.get("enabled")):
        return None
    base = Path(str(image_save.get("directory", "debug_policy_images"))).expanduser()
    if not base.is_absolute():
        base = Path.cwd() / base
    if bool(image_save.get("use_timestamp_subdir", True)):
        base = base / time.strftime("%Y%m%d_%H%M%S")
    base.mkdir(parents=True, exist_ok=True)
    cfg = {
        "dir": base,
        "interval": max(1, coerce_nonneg_int(image_save.get("interval"), 1)),
        "max_frames": max(0, coerce_nonneg_int(image_save.get("max_frames"), 0)),
        "saved": 0,
    }
    log_info(
        f"image_save: directory {base} (one PNG per camera after obs encode, before ZMQ send; "
        f"interval={cfg['interval']}, max_frames={cfg['max_frames']})"
    )
    return cfg


def _init_joints_dump_root(joints: dict[str, Any] | None, log_info) -> Path | None:
    if not joints or not bool(joints.get("enabled")):
        return None
    base = Path(str(joints.get("directory", "debug_joints"))).expanduser()
    if not base.is_absolute():
        base = Path.cwd() / base
    if bool(joints.get("use_timestamp_subdir", True)):
        base = base / time.strftime("%Y%m%d_%H%M%S")
    root = base / "client_pre_send"
    root.mkdir(parents=True, exist_ok=True)
    log_info(f"joints: saving arm_gripper_joints before ZMQ send under {root}")
    return root
