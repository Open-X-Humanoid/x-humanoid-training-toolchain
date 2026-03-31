#!/bin/bash
# Example: convert HDF5 data to LeRobot V3 dataset format
xhum-convert \
  --config /media/jushen/neil-liu/opensource/x-humanoid-training-toolchain/src/xhum/convert/configs/dvt217_stack_cube.json \
  --repo_id dvt217_stack_cube \
  --src_root /media/jushen/willwang/dataset/wholebody/tiangong20_dvt217_1rgb/dvt217_stack_cube_2026_0317/success_episodes \
  --tgt_path /media/jushen/neil-liu/dataNmodels/lerobot_dataset/dvt217/dvt217_stack_cube_2026_0317 \
  --task_name stack_cube
