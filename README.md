
# x-humanoid training toolchain (xhum)

[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Project Page](https://img.shields.io/badge/Project%20Page-RoboMIND-blue.svg)](https://x-humanoid-robomind.github.io/)
[![arXiv](https://badgen.net/badge/icon/arXiv?icon=awesome&label&color=red&style=flat-square)](https://arxiv.org/abs/2412.13877)
[![Dataset](https://img.shields.io/badge/Dataset-flopsera-000000.svg)](http://open.flopsera.com/flopsera-open/data-details/RoboMIND)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-RoboMIND-000000.svg)](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND)

Training and deployment toolchain for TienKung humanoid robots, built on [LeRobot](https://github.com/huggingface/lerobot) (included as a git submodule). The custom toolchain code (`xhum`) is fully decoupled from the upstream LeRobot codebase.

## Project Structure

```
x-humanoid-training-toolchain/
├── lerobot/                           # Git submodule -> huggingface/lerobot (v0.5.1)
├── src/xhum/                          # Custom toolchain (decoupled from lerobot)
│   ├── convert/
│   │   ├── hdf5_to_lerobot.py         # HDF5 -> LeRobot V3 dataset converter
│   │   ├── convert.sh                 # Conversion example script
│   │   └── configs/                   # Dataset conversion configs
│   ├── train/
│   │   └── configs/                   # Training configs (for lerobot-train)
│   ├── deploy/                        # Unified ROS2 deploy (./scripts/xhum-run xhum.deploy.ros2_deploy)
│   │   ├── policy_agent.py
│   │   ├── ros2_deploy.py
│   │   └── config.yaml
│   └── deploy_decouple/               # Py3.12 policy server + Py3.10 ROS ZMQ bridge (see README inside)
├── scripts/
│   └── xhum-run                       # Run xhum modules without pip install (sets PYTHONPATH=src)
├── pyproject.toml
└── Makefile
```

## Installation

### Prerequisites

- Python >= 3.12
- Git
- (Optional) CUDA-compatible GPU for training
- (Optional) ROS2 Humble/Iron for deployment

### Clone and install

```bash
git clone --recurse-submodules https://github.com/Open-X-Humanoid/x-humanoid-training-toolchain.git
cd x-humanoid-training-toolchain
make install          # LeRobot submodule only (same as make install-all)
```

Developer tools (formatting/tests) **without** installing `xhum`:

```bash
make install-dev      # LeRobot + pre-commit, pytest, ruff only
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

`make install` / `make install-all` install **LeRobot only** (from the submodule). Run `xhum` from the repo root with **`./scripts/xhum-run`** (sets `PYTHONPATH=src`) or `PYTHONPATH=src python -m …`.

### Verify installation

```bash
lerobot-train --help
./scripts/xhum-run xhum.convert.hdf5_to_lerobot --help
```

### Update LeRobot submodule

```bash
make update-lerobot
make install-lerobot
```

## Data Conversion

Convert HDF5 episode data into LeRobot V3 dataset format **without** installing `xhum`:

```bash
./scripts/xhum-run xhum.convert.hdf5_to_lerobot --help
```

### Source data layout

The converter expects a source directory containing episode subdirectories, each with an HDF5 file at a fixed relative path:

```
src_root/
├── episode_001/
│   └── data/trajectory.hdf5
├── episode_002/
│   └── data/trajectory.hdf5
└── episode_003/
    └── data/trajectory.hdf5
```

The relative path to the HDF5 file (default `data/trajectory.hdf5`) is configured via the `episode_path` field in the config JSON.

### Conversion config

A JSON config file defines the dataset metadata, output features, and HDF5-to-feature mappings. Example (`configs/dvt217_stack_cube.json`):

```json
{
    "dataset": {
        "fps": 30,
        "robot_type": "tienkung"
    },
    "episode_path": "data/trajectory.hdf5",
    "features": {
        "observation.state": {
            "dtype": "float32",
            "shape": [26],
            "names": null
        },
        "action": {
            "dtype": "float32",
            "shape": [16],
            "names": null
        },
        "observation.images.camera": {
            "dtype": "video",
            "shape": [360, 640, 3],
            "names": ["height", "width", "channels"]
        }
    },
    "mappings": [
        {
            "hdf5_key": "puppet/joint_position",
            "feature_key": "observation.state"
        },
        {
            "hdf5_key": "master/joint_position",
            "feature_key": "action"
        },
        {
            "hdf5_key": "observations/rgb_images/camera_camera",
            "feature_key": "observation.images.camera",
            "decode": "jpeg",
            "resize": [640, 360]
        }
    ]
}
```

**Config fields:**

| Field | Description |
|---|---|
| `dataset.fps` | Frame rate of the recorded data |
| `dataset.robot_type` | Robot identifier string |
| `episode_path` | Relative path from episode directory to HDF5 file |
| `features` | Output feature definitions (dtype, shape) following LeRobot V3 schema |
| `mappings[].hdf5_key` | Key path inside the HDF5 file |
| `mappings[].feature_key` | Corresponding output feature name |
| `mappings[].decode` | Set to `"jpeg"` / `"png"` / `"image"` for compressed image data |
| `mappings[].resize` | Optional `[width, height]` to resize decoded images |
| `mappings[].slice` | Optional `[start, end]` to slice array columns |

### Run conversion

```bash
./scripts/xhum-run xhum.convert.hdf5_to_lerobot \
  --config src/xhum/convert/configs/dvt217_stack_cube.json \
  --repo_id dvt217_stack_cube \
  --src_root /path/to/hdf5/episodes \
  --tgt_path /path/to/output \
  --task_name stack_cube
```

| Argument | Description |
|---|---|
| `--config` | Path to conversion config JSON |
| `--repo_id` | Output dataset name (created as `<tgt_path>/<repo_id>/`) |
| `--src_root` | Directory containing episode subdirectories |
| `--tgt_path` | Parent directory for the output dataset |
| `--task_name` | Task label stored with each frame (default: `default_task`) |

### Output format

The converted dataset follows LeRobot V3 layout:

```
<tgt_path>/<repo_id>/
├── meta/
│   ├── info.json              # Dataset metadata (fps, features, totals)
│   ├── stats.json             # Per-feature statistics
│   ├── tasks.parquet          # Task definitions
│   └── episodes/              # Per-episode metadata
├── data/
│   └── chunk-000/
│       └── file-000.parquet   # Numeric features (state, action, indices)
└── videos/
    └── observation.images.*/
        └── chunk-000/
            └── file-000.mp4   # Video data (multiple episodes per chunk)
```

Note: multiple episodes are stored in the same chunk file and distinguished by timestamp ranges in the episode metadata. This is normal LeRobot V3 behavior.

## Training

Train an ACT policy on a converted dataset using LeRobot's built-in CLI:

```bash
lerobot-train --config_path=src/xhum/train/configs/act_tienkung.json
```

The training config (`act_tienkung.json`) defines dataset, policy architecture, optimizer, and logging settings. Key fields to update before running:

| Field | Description |
|---|---|
| `dataset.repo_id` | Name of the converted dataset |
| `dataset.root` | Path to the dataset directory (or set `HF_LEROBOT_HOME` env var) |
| `output_dir` | Directory for checkpoints and logs |
| `policy.input_features` | Must match the features in your conversion config |
| `policy.output_features` | Action feature shape |

Checkpoints are saved in HuggingFace `from_pretrained`-compatible format, ready for deployment with `xhum.deploy.policy_agent.PolicyAgent` (or the decoupled copy under `src/xhum/deploy_decouple/algorithm/`).

## ROS2 deployment

From the repo root, unified ROS2 node (`hand_type` in YAML selects BrainCo vs Inspire):

```bash
./scripts/xhum-run xhum.deploy.ros2_deploy --config /path/to/src/xhum/deploy/config.yaml
```

For Python 3.12 policy + Python 3.10 ROS over ZMQ, see **`src/xhum/deploy_decouple/README.md`**.

## Related Projects

| Project | Description |
|---|---|
| [RoboMIND](https://github.com/x-humanoid-robomind/x-humanoid-robomind.github.io) | 107k real-world trajectories, 479 tasks, 96 object classes |
| [TienKung_URDF](https://github.com/x-humanoid-robomind/TienKung_URDF) | URDF package for ROS/Gazebo simulation |
| [TienKung_ROS](https://github.com/x-humanoid-robomind/TienKung_ROS) | Low-level ROS hardware control |
| [TienKung_Docs](https://github.com/x-humanoid-robomind/TienKung_Docs) | User manuals and SDK documentation |

## Acknowledgments

Built on top of [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face.
