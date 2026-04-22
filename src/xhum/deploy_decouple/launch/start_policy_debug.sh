#!/usr/bin/env bash
# Debug policy_server for the dvt217_stack_cube checkpoint.
# Pairs with robot/config/replay_debug.yaml (client side uses the same ../tmp/... dirs).
#
# Run order:
#   Terminal A (Py 3.12 + LeRobot):  bash launch/start_policy_debug.sh
#   Terminal B (Py 3.10):            cd robot && python3 run.py --config ./config/replay_debug.yaml
#
# After the run:
#   python3 scripts/compare_joints.py \
#     --client_dir tmp/debug_joint_dvt217/client_pre_send \
#     --server_dir tmp/debug_joint_dvt217/server_post_decode
#   diff -r tmp/debug_rgb_dvt217/client tmp/debug_rgb_dvt217/server

set -euo pipefail

# $ROOT = deploy_decouple/ ; resolves correctly no matter where the repo is cloned.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/policy"

# LEROBOT_SRC env var overrides; default assumes the toolchain submodule layout.
export PYTHONPATH="${LEROBOT_SRC:-$(cd "$ROOT/../../.." && pwd)/lerobot/src}:$PYTHONPATH"

# Override the model path via env when you switch checkpoints; keeps this file stable.
MODEL_PATH="${MODEL_PATH:-/media/jushen/neil-liu/dataNmodels/model_outputs/dvt217_run_002/checkpoints/200000/pretrained_model}"
TMP_ROOT="$ROOT/tmp"

# --joint_trace_flat + --save_images_flat: disable the auto timestamp subdir so
# client (writes to <TMP_ROOT>/debug_*/client*) and server (writes here under
# server_post_decode / server/) share a parent and compare cleanly.
exec python policy_server.py \
  --model_path "$MODEL_PATH" \
  --bind tcp://127.0.0.1:5555 \
  --joint_trace_dir "$TMP_ROOT/debug_joint_dvt217" \
  --joint_trace_flat \
  --save_images_dir "$TMP_ROOT/debug_rgb_dvt217/server" \
  --save_images_interval 20 \
  --save_images_max 10 \
  --save_images_flat
