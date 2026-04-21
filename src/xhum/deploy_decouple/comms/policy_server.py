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
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq

# This directory (comms) + sibling policy/ (PolicyAgent, LeRobot)
_ROOT = Path(__file__).resolve().parent
_DECOUPLE = _ROOT.parent
_POLICY = _DECOUPLE / "policy"
for _p in (_ROOT, _POLICY):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from policy_agent import PolicyAgent
from utils import maybe_save_obs_rgb_pngs, save_joints_vector
from zmq_obs_codec import multipart_to_obs


def _encode_action(tensor) -> tuple[bytes, bytes]:
    arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    meta = json.dumps({"version": 1, "shape": list(arr.shape)})
    return meta.encode("utf-8"), arr.tobytes()


def _server_log_error(msg: str) -> None:
    print(f"[policy_server] {msg}", flush=True)


def main():
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
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, args.linger)
    sock.bind(args.bind)
    print(f"[policy_server] bound {args.bind}", flush=True)

    agent = PolicyAgent(args.model_path)
    print("[policy_server] ready for requests", flush=True)

    joint_trace_root: Path | None = None
    if args.joint_trace_dir:
        jbase = Path(args.joint_trace_dir).expanduser()
        if not jbase.is_absolute():
            jbase = Path.cwd() / jbase
        if not args.joint_trace_flat:
            jbase = jbase / time.strftime("%Y%m%d_%H%M%S")
        joint_trace_root = jbase / "server_post_decode"
        joint_trace_root.mkdir(parents=True, exist_ok=True)
        print(f"[policy_server] joint_trace: saving decoded joints under {joint_trace_root}", flush=True)

    save_cfg: dict[str, Any] | None = None
    if args.save_images_dir:
        base = Path(args.save_images_dir).expanduser()
        if not base.is_absolute():
            base = Path.cwd() / base
        if not args.save_images_flat:
            base = base / time.strftime("%Y%m%d_%H%M%S")
        base.mkdir(parents=True, exist_ok=True)
        interval = max(1, int(args.save_images_interval))
        max_frames = max(0, int(args.save_images_max))
        save_cfg = {"dir": base, "interval": interval, "max_frames": max_frames, "saved": 0}
        print(
            f"[policy_server] save_images: {base} (after recv, before inference; "
            f"interval={interval}, max_frames={max_frames})",
            flush=True,
        )

    req_seq = 0
    while True:
        try:
            parts = sock.recv_multipart()
            obs = multipart_to_obs(parts)
            req_seq += 1
            if joint_trace_root is not None:
                save_joints_vector(
                    joint_trace_root,
                    req_seq,
                    obs["arm_gripper_joints"],
                    log_error=_server_log_error,
                )
            maybe_save_obs_rgb_pngs(save_cfg, obs, req_seq, log_error=_server_log_error)
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
