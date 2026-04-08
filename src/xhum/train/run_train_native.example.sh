#!/usr/bin/env bash
set -eu

# Native LeRobot training command (no install wrapper).
# Copy this file to run_train_native.sh and edit the values below.
#   cp run_train_native.example.sh run_train_native.sh
export HF_LEROBOT_HOME=/path/to/lerobot_dataset/
lerobot-train \
  --dataset.repo_id=your_user/your_dataset_folder/your_dataset \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir=/path/to/model_outputs/run_001 \
  --job_name=act_run_001 \
  --batch_size=32 \
  --steps=200000 \
  "$@"
