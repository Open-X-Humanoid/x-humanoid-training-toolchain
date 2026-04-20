#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load deploy_decouple ``PolicyAgent`` and run one inference with fake observations.

Requires Python >= 3.12, LeRobot, torch, opencv (same as ``policy_server.py``).

  cd src/xhum/deploy_decouple
  export PYTHONPATH=../../lerobot/src:algorithm   # from repo root adjust as needed
  python scripts/test_policy_agent_fake.py --model_path /path/to/pretrained_model

Or use the lerobot conda env (recommended):

  /opt/conda/envs/lerobot-0.5.1/bin/python scripts/test_policy_agent_fake.py --model_path ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# deploy_decouple/scripts -> deploy_decouple
_DEPLOY = Path(__file__).resolve().parents[1]
_ALGO = _DEPLOY / "algorithm"
_REPO_ROOT = _DEPLOY.parents[2]
_LEROBOT_SRC = _REPO_ROOT / "lerobot" / "src"

for p in (_ALGO, _LEROBOT_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from policy_agent import PolicyAgent


def _dims_from_config_json(model_dir: Path) -> tuple[dict[str, tuple[int, int, int]], int]:
    cfg_path = model_dir / "config.json"
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
        raise ValueError(f"could not parse input_features in {cfg_path}")
    return images, state_dim


def _make_fake_obs(images_meta: dict[str, tuple[int, int, int]], state_dim: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    obs_images: dict[str, np.ndarray] = {}
    for cam, (h, w, c) in images_meta.items():
        obs_images[cam] = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
    state = rng.standard_normal(state_dim).astype(np.float32) * 0.1
    return {"images": obs_images, "arm_gripper_joints": state}


def main() -> int:
    ap = argparse.ArgumentParser(description="PolicyAgent smoke test with fake obs")
    ap.add_argument(
        "--model_path",
        type=Path,
        default=Path("/media/jushen/neil-liu/dataNmodels/model_outputs/dvt217_run_002/checkpoints/last/pretrained_model"),
        help="Directory containing config.json and model weights",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    mp = args.model_path.resolve()
    if not mp.is_dir():
        print(f"ERROR: model_path is not a directory: {mp}", file=sys.stderr)
        return 1

    print(f"[fake] model_path={mp}")
    dims, state_dim = _dims_from_config_json(mp)
    print(f"[fake] cameras={list(dims.keys())}  state_dim={state_dim}")

    obs = _make_fake_obs(dims, state_dim, seed=args.seed)
    print("[fake] loading PolicyAgent …")
    agent = PolicyAgent(mp)
    agent.reset()
    print("[fake] inference …")
    with np.errstate(all="ignore"):
        action = agent.inference(obs)
    a = action.detach().cpu().numpy()
    print(f"[fake] OK  action shape={a.shape} dtype={a.dtype}")
    print(f"[fake]     min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
