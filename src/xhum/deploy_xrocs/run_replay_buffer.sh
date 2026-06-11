#!/usr/bin/env bash
# set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

# TODO: 修改以下路径
H5="/home/ubuntu/Dev/dylan_wu/data/h5_for_init_data/h5_data_for_init/station_sta1PlusH_dualArm-gripper-3cameras_72/tianshu_dualArm_72_grab_and_flip_label_up_white_bag_barcode_upward_20260530/trajectory.hdf5"
CONFIG="/home/ubuntu/Documents/configuration.toml"
REPO_ID="tianshu_dualArm_72_grab_and_flip_label_up_white_bag_barcode_upward_20260530"

python3 "/home/ubuntu/Dev/dylan_wu/project/infer/replay_buffer.py" \
  --h5 "${H5}" \
  --config "${CONFIG}" \
  --repo-id "${REPO_ID}" \
  --camera-view "left" \
  "$@"
