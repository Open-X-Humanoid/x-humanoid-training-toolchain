# Decoupled deployment (Python 3.12 policy + Python 3.10 ROS2)

**[中文说明](./README_zh.md)**

LeRobot / ACT inference needs **Python 3.12** (torch, `lerobot`). ROS2 on Ubuntu 22.04 typically uses **Python 3.10**. Mixing both in one process is fragile, so this package splits:

| Directory | Python | Role |
|-----------|--------|------|
| `algorithm/` | **3.12+** | `PolicyAgent` + **`policy_server.py`** — ZeroMQ **REP** |
| `ros_bridge/` | **3.10** (ROS) | Robot I/O + **`PolicyClient`** — ZeroMQ **REQ** |

The wire protocol is described in `algorithm/policy_server.py`.

---

## Repository layout

```
deploy_decouple/
├── README.md                 # This file (English)
├── README_zh.md              # Chinese
├── algorithm/
│   ├── policy_agent.py       # ACT wrapper (keep in sync with src/xhum/deployment/policy_agent.py)
│   ├── policy_server.py      # ZMQ server process
│   └── requirements.txt
├── ros_bridge/
│   ├── policy_client.py      # ZMQ client (numpy + pyzmq only)
│   ├── hdf5_actions.py       # HDF5 loaders for replay (joint_position or legacy *_align)
│   ├── ros2_node_zmq.py      # ROS2 node (model or replay)
│   ├── config_zmq.example.yaml   # Annotated example (copy → my_robot.yaml)
│   └── requirements.txt
└── scripts/
    ├── run_local_checks.py   # protocol | e2e
    ├── test_replay_hdf5.py   # Offline replay HDF5 load
    └── test_policy_agent_fake.py  # Load PolicyAgent + fake obs (no ROS)
```

---

## Same machine (typical): ROS + policy server on one PC

1. **Terminal A — policy server (Py 3.12, e.g. `conda activate lerobot-0.5.1`)**

```bash
cd src/xhum/deploy_decouple/algorithm
export PYTHONPATH=/path/to/x-humanoid-training-toolchain/lerobot/src:$PYTHONPATH
pip install pyzmq   # once, in this env

python policy_server.py \
  --model_path /path/to/checkpoints/last/pretrained_model \
  --bind tcp://127.0.0.1:5555
```

Use **`127.0.0.1`** so only local processes connect (simpler than `0.0.0.0` on a laptop).

2. **Terminal B — ROS (do *not* activate the lerobot conda env)**

```bash
source /opt/ros/humble/setup.bash
# source your workspace if needed

cd src/xhum/deploy_decouple/ros_bridge
pip install pyzmq pyyaml h5py numpy   # into the ROS Python if missing

cp config_zmq.example.yaml my_robot.yaml
# Set policy_server_url: tcp://127.0.0.1:5555 (must match --bind)
# If your checkpoint uses observation.images.camera, set obs_camera_key: camera (see YAML comments)

python3 ros2_node_zmq.py --config ./my_robot.yaml
```

**Order:** start **`policy_server` first**, then the ROS node.

---

## Configuration

- Copy **`ros_bridge/config_zmq.example.yaml`** → `my_robot.yaml`.
- Comments inside the YAML (Chinese) explain `mode`, `hand_type`, **same-machine ZMQ**, `replay` `h5_path`, `camera_name`, `action_rate`, and optional **`obs_camera_key`** (must match the short name in the model’s `observation.images.*`).

---

## Local tests (no robot / optional ROS)

From `src/xhum/deploy_decouple`:

| Command | Needs |
|---------|--------|
| `python scripts/run_local_checks.py protocol` | `numpy`, `pyzmq` — ZMQ roundtrip mock |
| `python scripts/run_local_checks.py e2e --model_path .../pretrained_model` | Py **≥3.12**, LeRobot, torch, pyzmq; sets `PYTHONPATH` for the server subprocess; uses `LEROBOT_PYTHON` if set, else tries `python3.12` or `/opt/conda/envs/lerobot-0.5.1/bin/python` |
| `python scripts/test_replay_hdf5.py /path/to/trajectory.hdf5` | `h5py`, `numpy` — same HDF5 logic as **`mode=replay`** |
| `python scripts/test_policy_agent_fake.py --model_path .../pretrained_model` | Py **≥3.12**, LeRobot, torch, opencv — loads **`algorithm/policy_agent.py`** once with random obs |

---

## `mode: replay`

Set `mode: replay` and `h5_path` to a **`trajectory.hdf5`** file. The node does **not** contact `policy_server`.

Supported HDF5 layouts (see `hdf5_actions.py`):

1. **`puppet/joint_position`** with shape `(T, 26)` — full command vector per step.
2. Legacy groups: `puppet/arm_*_align/data` and `end_effector_*_align/data`.

---

## Stay aligned with the monolithic stack

- Robot I/O and YAML behaviour mirror **`src/xhum/deploy/ros2_deploy.py`** (including replay HDF5 loading in that file after the same update).
- **`algorithm/policy_agent.py`** should stay in sync with **`src/xhum/deployment/policy_agent.py`** (obs dict: `images[<short_cam>]`, `arm_gripper_joints`).

---

## Security & ops

- ZMQ TCP is **not authenticated**. Prefer **`127.0.0.1`** on one machine; on a LAN use a firewall or SSH tunnel, e.g.  
  `ssh -L 5555:127.0.0.1:5555 user@policy-host`

## Optional: systemd

Run `policy_server.py` under **systemd** or **supervisor** so it restarts on failure; start `ros2_node_zmq.py` after the server is listening.
