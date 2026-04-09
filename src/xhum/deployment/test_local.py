#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local test script for ACT policy inference — no ROS2 required.

Usage examples:
    # Basic smoke test with the latest checkpoint
    python test_local.py

    # Point to a specific checkpoint
    python test_local.py --model_path /path/to/checkpoints/060000/pretrained_model

    # Use a real image for observation
    python test_local.py --image /path/to/test_image.jpg

    # Run latency benchmark (default 50 iterations)
    python test_local.py --benchmark --iterations 100

    # Compare outputs across all saved checkpoints
    python test_local.py --compare_checkpoints

    # Force CPU even when GPU is available
    python test_local.py --cpu
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))

from lerobot.policies.act.modeling_act import ACTPolicy

DEFAULT_MODEL_DIR = Path(
    "/media/jushen/neil-liu/dataNmodels/model_outputs/dvt217_run_002"
)
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "checkpoints" / "last" / "pretrained_model"

IMAGE_H, IMAGE_W = 360, 640
STATE_DIM = 26
ACTION_DIM = 16


# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────


def load_policy(model_path: Path, device: torch.device) -> ACTPolicy:
    print(f"[load] Loading model from: {model_path}")
    t0 = time.perf_counter()
    policy = ACTPolicy.from_pretrained(str(model_path))
    policy.eval()
    policy.to(device)
    elapsed = time.perf_counter() - t0
    print(f"[load] Model loaded in {elapsed:.2f}s on {device}")
    return policy


def make_random_obs(device: torch.device) -> dict:
    """Generate a random observation dict matching training data schema."""
    img = np.random.randint(0, 256, (IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    state = np.random.randn(STATE_DIM).astype(np.float32)
    state_tensor = torch.from_numpy(state)

    return {
        "observation.images.camera": img_tensor.unsqueeze(0).to(device),
        "observation.state": state_tensor.unsqueeze(0).to(device),
    }


def make_image_obs(image_path: str, device: torch.device) -> dict:
    """Build observation from a real image file + random state."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_W, IMAGE_H))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    state = np.zeros(STATE_DIM, dtype=np.float32)
    state_tensor = torch.from_numpy(state)

    return {
        "observation.images.camera": img_tensor.unsqueeze(0).to(device),
        "observation.state": state_tensor.unsqueeze(0).to(device),
    }


def run_inference(policy: ACTPolicy, obs: dict) -> torch.Tensor:
    with torch.inference_mode():
        action = policy.select_action(obs)
    return action


def print_separator(title: str = ""):
    width = 64
    if title:
        print(f"\n{'─' * 4} {title} {'─' * (width - len(title) - 6)}")
    else:
        print(f"{'─' * width}")


# ───────────────────────────────────────────────────────────────────
# Test routines
# ───────────────────────────────────────────────────────────────────


def test_smoke(policy: ACTPolicy, device: torch.device, image_path: str | None = None):
    """Run a single forward pass and validate output shape."""
    print_separator("Smoke Test")

    if image_path:
        print(f"[smoke] Using real image: {image_path}")
        obs = make_image_obs(image_path, device)
    else:
        print("[smoke] Using random observation")
        obs = make_random_obs(device)

    policy.reset()
    action = run_inference(policy, obs)

    print(f"[smoke] Action tensor shape : {list(action.shape)}")
    print(f"[smoke] Action dtype        : {action.dtype}")
    print(f"[smoke] Action device       : {action.device}")
    print(f"[smoke] Action range        : [{action.min().item():.4f}, {action.max().item():.4f}]")
    print(f"[smoke] Action mean / std   : {action.mean().item():.4f} / {action.std().item():.4f}")
    print(f"[smoke] First action step   : {action[0].cpu().numpy().round(4).tolist()}")

    expected_action_dim = ACTION_DIM
    actual_action_dim = action.shape[-1]
    if actual_action_dim != expected_action_dim:
        print(f"[smoke] WARNING: expected action dim {expected_action_dim}, got {actual_action_dim}")
    else:
        print(f"[smoke] PASS: action dim matches expected ({expected_action_dim})")

    return action


def test_benchmark(
    policy: ACTPolicy,
    device: torch.device,
    iterations: int = 50,
    image_path: str | None = None,
):
    """Measure inference latency over multiple iterations."""
    print_separator(f"Latency Benchmark ({iterations} iters)")

    if image_path:
        obs = make_image_obs(image_path, device)
    else:
        obs = make_random_obs(device)

    # warmup
    policy.reset()
    for _ in range(3):
        run_inference(policy, obs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    policy.reset()
    for i in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_inference(policy, obs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    lat = np.array(latencies) * 1000  # ms
    print(f"[bench] Mean   : {lat.mean():.2f} ms")
    print(f"[bench] Median : {np.median(lat):.2f} ms")
    print(f"[bench] Std    : {lat.std():.2f} ms")
    print(f"[bench] Min    : {lat.min():.2f} ms")
    print(f"[bench] Max    : {lat.max():.2f} ms")
    print(f"[bench] P95    : {np.percentile(lat, 95):.2f} ms")
    print(f"[bench] P99    : {np.percentile(lat, 99):.2f} ms")

    max_rate = 1000.0 / lat.mean()
    print(f"[bench] Throughput ≈ {max_rate:.1f} Hz  (target: 20 Hz)")
    if max_rate >= 20.0:
        print("[bench] PASS: inference is fast enough for 20 Hz deployment")
    else:
        print("[bench] WARNING: inference slower than 20 Hz target")


def test_determinism(policy: ACTPolicy, device: torch.device, trials: int = 3):
    """Verify that inference is deterministic given the same input."""
    print_separator("Determinism Check")

    torch.manual_seed(42)
    np.random.seed(42)
    obs = make_random_obs(device)

    results = []
    for i in range(trials):
        policy.reset()
        action = run_inference(policy, obs)
        results.append(action.cpu())

    all_same = all(torch.allclose(results[0], r, atol=1e-6) for r in results[1:])
    if all_same:
        print(f"[determ] PASS: {trials} runs produced identical outputs")
    else:
        diffs = [torch.abs(results[0] - r).max().item() for r in results[1:]]
        print(f"[determ] WARNING: outputs differ — max abs diff: {max(diffs):.6e}")


def test_multi_step(policy: ACTPolicy, device: torch.device, steps: int = 5):
    """Simulate a multi-step rollout to check temporal consistency."""
    print_separator(f"Multi-Step Rollout ({steps} steps)")

    policy.reset()
    actions = []
    for step in range(steps):
        obs = make_random_obs(device)
        action = run_inference(policy, obs)
        actions.append(action.cpu())
        print(
            f"  step {step}: action shape={list(action.shape)}  "
            f"range=[{action.min().item():.4f}, {action.max().item():.4f}]"
        )

    print(f"[multi] Completed {steps} inference steps without error")


def test_compare_checkpoints(device: torch.device, image_path: str | None = None):
    """Load each checkpoint and compare action outputs."""
    print_separator("Checkpoint Comparison")

    ckpt_root = DEFAULT_MODEL_DIR / "checkpoints"
    if not ckpt_root.exists():
        print(f"[compare] Checkpoint root not found: {ckpt_root}")
        return

    ckpt_dirs = sorted(
        [d for d in ckpt_root.iterdir() if d.is_dir() and d.name != "last"],
        key=lambda d: d.name,
    )

    if not ckpt_dirs:
        print("[compare] No checkpoint directories found")
        return

    print(f"[compare] Found {len(ckpt_dirs)} checkpoints: {[d.name for d in ckpt_dirs]}")

    torch.manual_seed(0)
    np.random.seed(0)
    if image_path:
        obs = make_image_obs(image_path, device)
    else:
        obs = make_random_obs(device)

    results = {}
    for ckpt_dir in ckpt_dirs:
        model_path = ckpt_dir / "pretrained_model"
        if not model_path.exists():
            print(f"  {ckpt_dir.name}: pretrained_model not found, skipping")
            continue

        try:
            policy = load_policy(model_path, device)
            policy.reset()
            action = run_inference(policy, obs)
            results[ckpt_dir.name] = action.cpu()
            a = action[0].cpu().numpy()
            print(
                f"  {ckpt_dir.name}: "
                f"mean={a.mean():.4f}  std={a.std():.4f}  "
                f"range=[{a.min():.4f}, {a.max():.4f}]"
            )
            del policy
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {ckpt_dir.name}: FAILED — {e}")

    if len(results) >= 2:
        keys = list(results.keys())
        first, last = keys[0], keys[-1]
        diff = torch.abs(results[first][0] - results[last][0]).mean().item()
        print(f"\n[compare] Mean action diff between {first} and {last}: {diff:.6f}")


def test_model_info(model_path: Path):
    """Print model config and weight statistics."""
    print_separator("Model Info")

    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            cfg = json.load(f)
        print(f"[info] Policy type       : {cfg.get('type', 'unknown')}")
        print(f"[info] Vision backbone   : {cfg.get('vision_backbone', 'unknown')}")
        print(f"[info] Chunk size        : {cfg.get('chunk_size', 'unknown')}")
        print(f"[info] n_action_steps    : {cfg.get('n_action_steps', 'unknown')}")
        print(f"[info] Latent dim (VAE)  : {cfg.get('latent_dim', 'unknown')}")
        print(f"[info] Transformer dim   : {cfg.get('dim_model', 'unknown')}")

        for key, feat in cfg.get("input_features", {}).items():
            print(f"[info] Input  {key}: {feat['shape']}")
        for key, feat in cfg.get("output_features", {}).items():
            print(f"[info] Output {key}: {feat['shape']}")
    else:
        print(f"[info] config.json not found at {config_file}")

    train_cfg_file = model_path / "train_config.json"
    if train_cfg_file.exists():
        with open(train_cfg_file) as f:
            tcfg = json.load(f)
        print(f"[info] Training steps    : {tcfg.get('steps', 'unknown')}")
        print(f"[info] Batch size        : {tcfg.get('batch_size', 'unknown')}")
        print(f"[info] Dataset           : {tcfg.get('dataset', {}).get('repo_id', 'unknown')}")
        print(f"[info] Learning rate     : {tcfg.get('policy', {}).get('optimizer_lr', 'unknown')}")

    safetensors_file = model_path / "model.safetensors"
    if safetensors_file.exists():
        size_mb = safetensors_file.stat().st_size / (1024 * 1024)
        print(f"[info] Weights file size : {size_mb:.1f} MB")


# ───────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local test script for ACT policy inference (no ROS2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to pretrained_model directory",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a real image file for observation",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--compare_checkpoints",
        action="store_true",
        help="Compare action outputs across all saved checkpoints",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even when GPU is available",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"VRAM  : {mem:.1f} GB")

    print_separator("Configuration")
    print(f"  model_path : {model_path}")
    print(f"  image      : {args.image or '(random)'}")
    print(f"  seed       : {args.seed}")

    if not model_path.exists():
        print(f"\nERROR: model path does not exist: {model_path}")
        sys.exit(1)

    test_model_info(model_path)

    if args.compare_checkpoints:
        test_compare_checkpoints(device, image_path=args.image)
        return

    policy = load_policy(model_path, device)

    test_smoke(policy, device, image_path=args.image)
    test_determinism(policy, device)
    test_multi_step(policy, device)

    if args.benchmark:
        test_benchmark(policy, device, iterations=args.iterations, image_path=args.image)

    print_separator()
    print("All tests completed.")


if __name__ == "__main__":
    main()
