#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local checks for deploy_decouple (ZMQ client ↔ server).

  # No model / no LeRobot — verifies multipart protocol + PolicyClient
  python scripts/run_local_checks.py protocol

  # Needs Py3.12+ env with LeRobot + checkpoint (same as policy_server.py)
  python scripts/run_local_checks.py e2e --model_path /path/to/pretrained_model

Run from ``src/xhum/deploy_decouple`` (this directory's parent is ``deploy_decouple``):

  cd src/xhum/deploy_decouple
  pip install numpy pyzmq
  python scripts/run_local_checks.py protocol
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zmq

# deploy_decouple/scripts -> deploy_decouple
_DEPLOY = Path(__file__).resolve().parents[1]
_REPO_ROOT = _DEPLOY.parents[2]
_LEROBOT_SRC = _REPO_ROOT / "lerobot" / "src"
_ROS_BRIDGE = _DEPLOY / "ros_bridge"
_ALGO = _DEPLOY / "algorithm"
if str(_ROS_BRIDGE) not in sys.path:
    sys.path.insert(0, str(_ROS_BRIDGE))

from policy_client import PolicyClient


def _free_tcp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _decode_obs_like_server(parts: list[bytes]) -> dict:
    """Mirror algorithm/policy_server._decode_obs for assertions."""
    meta = json.loads(parts[0].decode("utf-8"))
    state = np.asarray(meta["state"], dtype=np.float32)
    obs_images: dict[str, np.ndarray] = {}
    idx = 1
    for cam_name, shape in meta["images"].items():
        h, w, c = int(shape[0]), int(shape[1]), int(shape[2])
        buf = parts[idx]
        idx += 1
        arr = np.frombuffer(memoryview(buf), dtype=np.uint8).reshape(h, w, c)
        obs_images[cam_name] = arr
    return {"images": obs_images, "arm_gripper_joints": state}


def _mock_policy_server_thread(endpoint: str, barrier: threading.Barrier, errors: list[str]):
    """REP socket: verify request, reply with deterministic float32 action."""
    try:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(endpoint)
        barrier.wait(timeout=5)

        parts = sock.recv_multipart()
        obs = _decode_obs_like_server(parts)

        img = obs["images"]["camera"]
        if img.shape != (60, 80, 3):
            errors.append(f"bad image shape {img.shape}")
        if not np.allclose(obs["arm_gripper_joints"], np.arange(8, dtype=np.float32)):
            errors.append("state mismatch")

        action = np.arange(16, dtype=np.float32).reshape(1, 16)
        meta = json.dumps({"version": 1, "shape": list(action.shape)})
        sock.send_multipart([meta.encode("utf-8"), action.tobytes(order="C")])
        sock.close(0)
        ctx.term()
    except Exception as e:
        errors.append(str(e))


def run_protocol_test() -> int:
    print("[protocol] ZMQ mock server + PolicyClient roundtrip …")
    port = _free_tcp_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    errors: list[str] = []
    barrier = threading.Barrier(2, timeout=10)
    th = threading.Thread(target=_mock_policy_server_thread, args=(endpoint, barrier, errors), daemon=True)
    th.start()
    try:
        barrier.wait(timeout=10)
    except threading.BrokenBarrierError:
        print("[protocol] FAIL: server thread did not start", file=sys.stderr)
        return 1

    time.sleep(0.05)

    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img[:, :, 0] = np.arange(60 * 80, dtype=np.uint8).reshape(60, 80) % 251
    state = np.arange(8, dtype=np.float32)
    obs = {"images": {"camera": img}, "arm_gripper_joints": state}

    try:
        with PolicyClient(server_url=endpoint, timeout_ms=5000) as cli:
            out = cli.inference(obs)
    except Exception as e:
        print(f"[protocol] FAIL: client error: {e}", file=sys.stderr)
        th.join(timeout=2)
        return 1

    th.join(timeout=2)
    if errors:
        print(f"[protocol] FAIL: server errors: {errors}", file=sys.stderr)
        return 1

    if out.shape != (1, 16) or not np.allclose(out, np.arange(16, dtype=np.float32).reshape(1, 16)):
        print(f"[protocol] FAIL: bad action {out.shape} {out}", file=sys.stderr)
        return 1

    print("[protocol] OK")
    return 0


def _dims_from_pretrained_dir(model_dir: Path) -> tuple[dict[str, tuple[int, int, int]], int]:
    """Return (camera_short_name -> (H,W,C), state_dim) from config.json."""
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    images: dict[str, tuple[int, int, int]] = {}
    state_dim = 0
    for key, feat in cfg.get("input_features", {}).items():
        t = str(feat.get("type", "")).lower()
        shape = feat.get("shape") or []
        if "visual" in t or t == "image":
            if len(shape) == 3:
                c, h, w = int(shape[0]), int(shape[1]), int(shape[2])
                short = key.rsplit(".", maxsplit=1)[-1]
                images[short] = (h, w, c)
        elif "state" in t:
            if isinstance(shape, list) and len(shape) >= 1:
                state_dim = int(shape[0])
    if not images or state_dim <= 0:
        raise ValueError(f"could not parse input_features from {cfg_path}")
    return images, state_dim


def _pick_policy_server_python() -> str:
    """LeRobot in this repo expects Python >= 3.12 (syntax). Override with LEROBOT_PYTHON."""
    envp = os.environ.get("LEROBOT_PYTHON")
    if envp and Path(envp).is_file():
        return envp
    candidates = [
        shutil.which("python3.12"),
        "/opt/conda/envs/lerobot-0.5.1/bin/python",
        sys.executable,
    ]
    for exe in candidates:
        if not exe or not Path(exe).is_file():
            continue
        try:
            out = subprocess.run(
                [exe, "-c", "import sys; assert sys.version_info >= (3, 12); print('ok')"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0:
                return exe
        except (OSError, subprocess.TimeoutExpired):
            continue
    return sys.executable


def _wait_tcp(host: str, port: int, timeout_s: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=1.0)
            s.close()
            return True
        except OSError:
            time.sleep(0.5)
    return False


def run_e2e_test(model_path: Path) -> int:
    print(f"[e2e] model_path={model_path}")
    if not model_path.is_dir():
        print(f"[e2e] FAIL: not a directory: {model_path}", file=sys.stderr)
        return 1

    port = _free_tcp_port()
    bind = f"tcp://127.0.0.1:{port}"
    py = _pick_policy_server_python()
    print(f"[e2e] policy_server python: {py}")
    server_py = _ALGO / "policy_server.py"
    env = os.environ.copy()
    pp = os.pathsep.join(
        p
        for p in (str(_ALGO), str(_LEROBOT_SRC), env.get("PYTHONPATH", ""))
        if p
    )
    env["PYTHONPATH"] = pp

    proc = subprocess.Popen(
        [py, str(server_py), "--model_path", str(model_path), "--bind", bind],
        cwd=str(_ALGO),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        if not _wait_tcp("127.0.0.1", port, timeout_s=120.0):
            err = b""
            if proc.stderr:
                err = proc.stderr.read()
            rc = proc.poll()
            print(
                f"[e2e] FAIL: server TCP not accepting (timeout). proc_returncode={rc}",
                file=sys.stderr,
            )
            if err.strip():
                print(err.decode("utf-8", errors="replace")[-8000:], file=sys.stderr)
            return 1

        time.sleep(2.0)

        dims, state_dim = _dims_from_pretrained_dir(model_path)
        if len(dims) != 1:
            print(f"[e2e] WARN: expected 1 camera in config, got {list(dims.keys())}; using first only")

        (cam, (h, w, c)) = next(iter(dims.items()))
        img = (np.random.randint(0, 255, (h, w, c), dtype=np.uint8) // 4 * 4).astype(np.uint8)
        state = np.zeros(state_dim, dtype=np.float32)

        obs = {"images": {cam: img}, "arm_gripper_joints": state}
        url = f"tcp://127.0.0.1:{port}"

        t0 = time.perf_counter()
        with PolicyClient(server_url=url, timeout_ms=60_000) as cli:
            out = cli.inference(obs)
        dt = time.perf_counter() - t0
        print(f"[e2e] inference wall time: {dt:.2f}s  action shape={out.shape}")

        if out.ndim != 2 or out.shape[0] != 1:
            print(f"[e2e] FAIL: unexpected action shape {out.shape}", file=sys.stderr)
            return 1

        print("[e2e] OK")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def main() -> int:
    ap = argparse.ArgumentParser(description="Local deploy_decouple checks")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("protocol", help="ZMQ roundtrip without LeRobot (numpy + pyzmq)")
    p1 = sub.add_parser("e2e", help="Start policy_server + one real inference (needs model)")
    p1.add_argument("--model_path", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "protocol":
        return run_protocol_test()
    if args.cmd == "e2e":
        return run_e2e_test(args.model_path.resolve())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
