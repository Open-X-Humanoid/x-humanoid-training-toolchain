#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remote policy server (Python 3.12 + LeRobot).

ROS / Python 3.10 nodes talk to this process over ZeroMQ REQ/REP — no torch
inside the ROS interpreter.

Protocol (multipart ZMQ):
  Request (inference):  [meta_json_utf8, img_bytes_0, img_bytes_1, ...]
    meta JSON: {
      "version": 1,
      "op": "infer",                                    # optional; default "infer"
      "state": [float, ...],
      "images": {"<short_cam_name>": [H, W, 3], ...}    # order matches following frames
    }
    Each image: row-major uint8 RGB, shape as in meta.

  Request (reset):      [meta_json_utf8]
    meta JSON: {"version": 1, "op": "reset"}

  Reply (inference):    [meta_json_utf8, action_float32_bytes]
    meta: {"version": 1, "op": "infer", "shape": [1, action_dim]}
  Reply (reset):        [meta_json_utf8, b""]
    meta: {"version": 1, "op": "reset_ack"}
  Reply (error):        [meta_json_utf8, b""]
    meta: {"version": 1, "error": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq

# This file runs in the Py 3.12 env. PolicyAgent is a sibling module; the wire/
# codec + trace helpers live one level up in ``deploy_decouple/wire`` — no pip
# install required, each process injects it at startup.
_HERE = Path(__file__).resolve().parent
_WIRE = _HERE.parent / "wire"
for _p in (_HERE, _WIRE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from policy_agent import PolicyAgent
from trace_io import maybe_save_obs_rgb_pngs, save_joints_vector
from obs_codec import OP_INFER, OP_RESET, PROTOCOL_VERSION, multipart_to_meta, multipart_to_obs


def _encode_action_reply(tensor) -> tuple[bytes, bytes]:
    arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    meta = {"version": PROTOCOL_VERSION, "op": OP_INFER, "shape": list(arr.shape)}
    return json.dumps(meta).encode("utf-8"), arr.tobytes()


def _encode_reset_ack() -> tuple[bytes, bytes]:
    meta = {"version": PROTOCOL_VERSION, "op": "reset_ack"}
    return json.dumps(meta).encode("utf-8"), b""


def _encode_error_reply(err: str) -> tuple[bytes, bytes]:
    meta = {"version": PROTOCOL_VERSION, "error": err, "shape": []}
    return json.dumps(meta).encode("utf-8"), b""


def _server_log_error(msg: str) -> None:
    print(f"[policy_server] {msg}", flush=True)


def _build_joint_trace_dir(raw: str | None, flat: bool) -> Path | None:
    if not raw:
        return None
    base = Path(raw).expanduser()
    if not base.is_absolute():
        base = Path.cwd() / base
    if not flat:
        base = base / time.strftime("%Y%m%d_%H%M%S")
    out = base / "server_post_decode"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_image_save_cfg(raw_dir: str | None, flat: bool, interval: int, max_frames: int) -> dict[str, Any] | None:
    if not raw_dir:
        return None
    base = Path(raw_dir).expanduser()
    if not base.is_absolute():
        base = Path.cwd() / base
    if not flat:
        base = base / time.strftime("%Y%m%d_%H%M%S")
    base.mkdir(parents=True, exist_ok=True)
    return {
        "dir": base,
        "interval": max(1, int(interval)),
        "max_frames": max(0, int(max_frames)),
        "saved": 0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZMQ policy server (Py3.12 + LeRobot)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained_model directory")
    parser.add_argument("--bind", type=str, default="tcp://0.0.0.0:5555", help="ZMQ bind address")
    parser.add_argument("--linger", type=int, default=0, help="ZMQ socket linger (ms)")
    parser.add_argument(
        "--save_images_dir",
        type=str,
        default=None,
        help="If set, save decoded RGB PNGs here after each receive, before inference (same layout as YAML image_save).",
    )
    parser.add_argument("--save_images_interval", type=int, default=1, help="Save every N-th request (>=1)")
    parser.add_argument(
        "--save_images_max",
        type=int,
        default=0,
        help="Stop saving after this many saved steps (0 = unlimited)",
    )
    parser.add_argument(
        "--save_images_flat",
        action="store_true",
        help="Write directly under save_images_dir (default: create YYYYMMDD_HHMMSS subdir)",
    )
    parser.add_argument(
        "--joint_trace_dir",
        type=str,
        default=None,
        help="If set, save decoded arm_gripper_joints after each recv under .../server_post_decode/state_*.npy",
    )
    parser.add_argument(
        "--joint_trace_flat",
        action="store_true",
        help="Write server_post_decode directly under joint_trace_dir (no timestamp subdir)",
    )
    return parser.parse_args()


def _bind_rep_socket(ctx: zmq.Context, bind: str, linger: int) -> zmq.Socket:
    """Create + bind a fresh REP socket. Called at startup and after any ZMQ error
    to recover from a stuck REP state machine (see main loop below)."""
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, linger)
    sock.bind(bind)
    return sock


def main():
    args = _parse_args()

    ctx = zmq.Context()
    sock = _bind_rep_socket(ctx, args.bind, args.linger)
    print(f"[policy_server] bound {args.bind}", flush=True)

    # Loading LeRobot weights can take seconds; print readiness only after.
    agent = PolicyAgent(args.model_path)
    print("[policy_server] ready for requests", flush=True)

    joint_trace_root = _build_joint_trace_dir(args.joint_trace_dir, args.joint_trace_flat)
    if joint_trace_root is not None:
        print(f"[policy_server] joint_trace: saving decoded joints under {joint_trace_root}", flush=True)

    save_cfg = _build_image_save_cfg(
        args.save_images_dir,
        args.save_images_flat,
        args.save_images_interval,
        args.save_images_max,
    )
    if save_cfg is not None:
        print(
            f"[policy_server] save_images: {save_cfg['dir']} (after recv, before inference; "
            f"interval={save_cfg['interval']}, max_frames={save_cfg['max_frames']})",
            flush=True,
        )

    # REP state machine: recv → send → recv → send. Any ZMQ error stalls the
    # socket; the only safe recovery is close + re-bind. We do this both on
    # recv and send failure rather than letting the loop spin on a dead socket.
    infer_seq = 0
    while True:
        try:
            parts = sock.recv_multipart()
        except zmq.ZMQError as e:
            _server_log_error(f"recv failed ({e}); rebinding REP socket")
            try:
                sock.close(linger=0)
            except Exception:
                pass
            sock = _bind_rep_socket(ctx, args.bind, args.linger)
            continue

        # Single try-block: any failure in parsing, decoding, or inference
        # must still produce a reply — REP spec requires send-after-recv.
        reply_meta: bytes
        reply_body: bytes
        try:
            meta = multipart_to_meta(parts)
            # ``op`` defaults to infer for backward compat with older clients
            # that were written before the reset frame existed.
            op = meta.get("op", OP_INFER)
            if op == OP_RESET:
                # Clears ACT's internal action buffer so the next episode
                # doesn't inherit half-consumed chunks from the previous one.
                agent.reset()
                print(f"[policy_server] reset acknowledged (infer_seq={infer_seq})", flush=True)
                reply_meta, reply_body = _encode_reset_ack()
            elif op == OP_INFER:
                obs = multipart_to_obs(parts, meta)
                infer_seq += 1
                # Dumps happen **after decode, before inference** so the
                # bytes on disk match exactly what the model will see — key
                # invariant for client/server calibration (scripts/compare_joints).
                if joint_trace_root is not None:
                    save_joints_vector(
                        joint_trace_root, infer_seq, obs["arm_gripper_joints"], log_error=_server_log_error
                    )
                maybe_save_obs_rgb_pngs(save_cfg, obs, infer_seq, log_error=_server_log_error)
                out = agent.inference(obs)
                reply_meta, reply_body = _encode_action_reply(out)
            else:
                raise ValueError(f"unknown op: {op!r}")
        except Exception as e:
            # Errors become reply meta (not exceptions to the client's
            # sock.recv), keeping the REP state machine intact.
            reply_meta, reply_body = _encode_error_reply(str(e))
            print(f"[policy_server] error: {e}", flush=True)

        try:
            sock.send_multipart([reply_meta, reply_body])
        except zmq.ZMQError as e:
            _server_log_error(f"send failed ({e}); rebinding REP socket")
            try:
                sock.close(linger=0)
            except Exception:
                pass
            sock = _bind_rep_socket(ctx, args.bind, args.linger)


if __name__ == "__main__":
    main()
