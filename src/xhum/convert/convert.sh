#!/bin/bash
# Example: convert HDF5 data to LeRobot V3 dataset format
xhum-convert \
  --config /home/neil/workspace/claude-toolchain-try/x-humanoid/src/xhum/convert/configs/dvt217_stack_cube.json \
  --repo_id dvt217_stack_cube \
  --src_root /home/neil/workspace/data/sub_dvt217_stack_cube_2026_0317/success_episodes \
  --tgt_path /home/neil/workspace/data/lerobot_output \
  --task_name stack_cube
