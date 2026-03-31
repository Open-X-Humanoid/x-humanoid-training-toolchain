#!/usr/bin/env bash
set -euo pipefail

# Native LeRobot training command (no install wrapper).
# Edit the values below as needed.

lerobot-train \
  --dataset.repo_id=dvt217/dvt217_stack_cube_2026_0317/dvt217_stack_cube \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir=/media/jushen/neil-liu/dataNmodels/model_outputs/dvt217_run_002 \
  --job_name=act_run_002 \
  --batch_size=16 \
  --steps=40000 \
  "$@"
