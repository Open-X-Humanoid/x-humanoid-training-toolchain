#!/usr/bin/env bash
# Multi-dataset π₀.₅ finetuning via xhum.train.train_multi (MEAN_STD — no quantiles).
#
# Prerequisites:
#   cd lerobot && pip install -e ".[pi]" && cd ..
#   pip install wandb && export WANDB_API_KEY=...
#   make install
# WandB: edit wandb.project / training.job_name in the JSON config
#
# Configs (real paths under /media/users/wd/data/...):
#   - white_bag x2: multi_train_pi05_tianshu_72_white_bag_barcode_upward_260602.json
#   - all 4 tasks:  multi_train_pi05_tianshu_72.json
#
# Usage (from repo root):
#   bash src/xhum/train/run_train_pi05_multi.example.sh
#   bash multi_train_pi05_tianshu_72_white_bag_barcode_upward_260602.sh

set -eu

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

CONFIG="src/xhum/train/configs/multi_train_pi05_tianshu_73_new_white_upward_downward_unfreeze_ve_0611.json"

# Multi-GPU launch (8 GPUs). accelerate launch must invoke Python, not ./scripts/xhum-run (bash).
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --module xhum.train.train_multi \
  -- --config "${CONFIG}" "$@"
