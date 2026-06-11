#!/usr/bin/env bash
# Convert station_sta1PlusH_dualArm-gripper-3cameras_72 HDF5 → LeRobot V3.
#
# Usage (from repository root):
#   1. Edit TASKS below: fill in task_name (natural-language instruction) for each task_dir
#   2. bash src/xhum/convert/tianshu_72_convert.example.sh
#
# Convert a single task:
#   bash src/xhum/convert/tianshu_72_convert.example.sh tianshu_dualArm_72_grab_and_flip_label_up_white_bag_barcode_upward
#
# Edit H5_ROOT / TGT_PATH / CONFIG below if your layout differs.


ROOT="/media/users/wd/projects/demo/x-humanoid-training-toolchain"
DATA_ROOT="/media/jushen/xr-2/wd/data"


CONFIG="${ROOT}/src/xhum/convert/configs/tianshu_73_express_demo_pi05_puppet.json"
H5_ROOT="${DATA_ROOT}/station_data/h5_data/station_sta1PlusH_dualArm-gripper-3cameras_73"
TGT_PATH="${DATA_ROOT}/station_data/lerobot_v3/station_sta1PlusH_dualArm-gripper-3cameras_73"
DECODE_WORKERS="${DECODE_WORKERS:-15}"
# task_dir (under H5_ROOT) | task_name → LeRobot meta/tasks.parquet 中的 instruction
# repo_id 与 task_dir 相同；task_name 需手动填写，不要用目录名。

insturction="The left arm picks the parcel from the recess, places it on the table and checks the tracking number. Flip it if the number faces down, then the right arm pushes the parcel with the number facing up onto the conveyor belt."


TASKS=(
#    "tianshu_dualArm_73_grab_and_flip_label_up_white_bag_barcode_upward_20260603|$insturction"
#    "tianshu_dualArm_73_grab_and_flip_label_up_black_bag_barcode_upward|$insturction"
#    "tianshu_dualArm_73_grab_and_flip_label_up_black_bag_barcode_upward_20260604|$insturction"
#    "tianshu_dualArm_73_grab_and_flip_label_up_white_bag_barcode_downward|$insturction"
    # "tianshu_dualArm_73_grab_and_flip_label_up_white_bag_barcode_downward_20260605|$insturction"
    # "tianshu_dualArm_73_grab_and_flip_label_up_black_bag_barcode_downward|$insturction"
    # "tianshu_dualArm_73_stack_grab_and_flip_label_up_white_bag_barcode_upward|$insturction"
    # "tianshu_dualArm_73_stack_grab_and_flip_label_up_white_bag_barcode_upward_20260607|$insturction"
    # "tianshu_dualArm_73_stack_grab_and_flip_label_up_black_bag_barcode_upward|$insturction"
    "tianshu_dualArm_73_grab_and_flip_label_up_white_bag_barcode_upward_20260610|$insturction"
    "tianshu_dualArm_73_grab_and_flip_label_up_white_bag_barcode_downward_20260610|$insturction"
)


run_convert() {
    local task_dir="$1"
    local task_name="$2"
    local src_root="${H5_ROOT}/${task_dir}/success_episodes"

    if [[ -z "$task_name" || "$task_name" == TODO* ]]; then
        echo "ERROR: task_name not set for $task_dir — edit TASKS in this script." >&2
        exit 1
    fi

    if [[ ! -d "$src_root" ]]; then
        echo "SKIP: success_episodes not found: $src_root" >&2
        return 1
    fi

    echo "================================================================"
    echo "Converting: $task_dir"
    echo "  src_root : $src_root"
    echo "  repo_id  : $task_dir"
    echo "  task_name: $task_name"
    echo "  tgt_path : $TGT_PATH"
    echo "================================================================"

    "${ROOT}/scripts/xhum-run" xhum.convert.hdf5_to_lerobot \
        --config "$CONFIG" \
        --repo_id "$task_dir" \
        --src_root "$src_root" \
        --tgt_path "$TGT_PATH" \
        --task_name "$task_name" \
        --decode-workers "$DECODE_WORKERS"
}


run_task() {
    local task_dir="$1"
    for entry in "${TASKS[@]}"; do
        IFS='|' read -r dir name <<< "$entry"
        if [[ "$dir" == "$task_dir" ]]; then
            run_convert "$dir" "$name"
            return
        fi
    done
    echo "Unknown task: $task_dir" >&2
    echo "Available tasks:" >&2
    for entry in "${TASKS[@]}"; do
        IFS='|' read -r dir _ <<< "$entry"
        echo "  $dir" >&2
    done
    exit 1
}

if [[ $# -gt 0 ]]; then
    run_task "$1"
else
    for entry in "${TASKS[@]}"; do
        IFS='|' read -r task_dir task_name <<< "$entry"
        run_convert "$task_dir" "$task_name"
    done
fi

echo "Done. Output under: ${TGT_PATH}/"
