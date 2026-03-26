#!/bin/bash
# Example: convert HDF5 data to LeRobot V3 dataset format
xhum-convert \
  --config src/xhum/convert/configs/Tien_Kung_Gello_1RGB.json \
  --repo_id EXAMPLE \
  --src_root PATH_TO_ROOT \
  --tgt_path PATH_TO_TARGET \
  --task_name EXAMPLE \
  --fps 30 \
  --robot_type tienkung
