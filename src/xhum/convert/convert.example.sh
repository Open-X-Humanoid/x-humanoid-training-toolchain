#!/bin/bash
# Example: convert HDF5 data to LeRobot V3 dataset format.
# Copy this file to convert.sh and edit the paths below.
#   cp convert.example.sh convert.sh
xhum-convert \
  --config /path/to/configs/your_config.json \
  --repo_id your_dataset \
  --src_root /path/to/source/episodes \
  --tgt_path /path/to/lerobot_dataset/output \
  --task_name your_task
