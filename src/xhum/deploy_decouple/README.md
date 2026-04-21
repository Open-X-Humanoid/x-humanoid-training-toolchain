# Decoupled deployment (Python 3.12 policy + Python 3.10 ROS2)

**[中文说明](./README_zh.md)**

LeRobot / ACT inference needs **Python 3.12** (torch, `lerobot`). ROS2 on Ubuntu 22.04 typically uses **Python 3.10**. Mixing both in one process is fragile, so this package splits:

| Directory | Python | Role |
|-----------|--------|------|
| `policy/` | **3.12+** | `PolicyAgent` only (LeRobot / ACT); imported by the policy server |
| `comms/` | **3.12+** (server) / **3.10** (client) | **`policy_server.py`**, **`policy_client.py`**, **`zmq_obs_codec.py`**, **`utils.py`** — ZMQ wire + processes |
| `robot/` | **3.10** (ROS) | Robot + HDF5 I/O; loads **`PolicyClient`** from `comms/` when **`mode=model`** or **`mode=replay`** |

**Why client on ROS:** the control loop runs in the ROS process and *pulls* one action per step from the policy process, so the ROS side is the natural **request** initiator. The policy process **binds** and **replies** with inference results.

The wire protocol is described in `comms/policy_server.py`.

**Model path:** only when starting **`comms/policy_server.py`** via **`--model_path`**. **Do not put `model_path` in `robot/*.yaml`** — the ZMQ wire carries observations and actions only; the ROS client neither sends nor needs the checkpoint directory. To switch checkpoints, **restart the policy server** with a new **`--model_path`**.

---

## Repository layout

```
deploy_decouple/
├── README.md                 # This file (English)
├── README_zh.md              # Chinese
├── policy/
│   ├── policy_agent.py       # ACT wrapper (keep in sync with src/xhum/deploy/policy_agent.py)
│   └── requirements.txt
├── comms/
│   ├── policy_server.py      # ZMQ REP server (Py3.12 + LeRobot; imports PolicyAgent)
│   ├── policy_client.py      # ZMQ REQ client (numpy + pyzmq); model / replay / replay_debug
│   ├── zmq_obs_codec.py      # Shared obs multipart encode/decode
│   └── utils.py              # PNG / YAML int helpers (client + server)
├── robot/
│   ├── replay_io/            # HDF5 loaders (actions + RGB/state for ZMQ replay)
│   ├── settings/             # YAML merge + PolicyClient factory (no ROS)
│   ├── config/               # Example YAML (copy to robot/ root or pass path)
│   ├── ros2_node_zmq.py      # Node implementation (also runnable)
│   ├── run.py                # Thin entry — model / replay / replay_actions (ROS2) or replay_debug (no ROS)
│   └── requirements.txt
├── launch/                   # Example shell wrappers (see launch/README.md)
│   ├── start_policy.example.sh
│   └── start_robot.example.sh
└── scripts/
    ├── README.md                      # Script usage
    └── stat_hdf5_firstframe_mean.py   # Mean of dataset[key][0] over many HDF5 (tracked); other *.py: see README
```

---

## Updates (branch `refactor/decouple-toolchain`)

- **Layout:** former `algorithm/`, `ros_bridge/`, etc. moved under `policy/`, `comms/`, `robot/`, `launch/`; use this README’s tree as the source of truth for entrypoints.
- **PolicyAgent:** if `policy_preprocessor.json` and `policy_postprocessor.json` sit next to the checkpoint, inference matches LeRobot `predict_action` (normalize observations, **denormalize actions**). If those files are missing, behavior falls back to raw `select_action` (no denorm).
- **HDF5 eval / local scripts:** `*.py` under `scripts/` are not tracked; commands and examples live in **[`scripts/README.md`](./scripts/README.md)**.
- **`src/xhum/deploy/policy_agent.py`** stays in sync with `deploy_decouple/policy/policy_agent.py` for the same preprocessor/postprocessor wiring.

---

## Same machine: four YAML **`mode`** values (`model` / `replay` / `replay_actions` / `replay_debug`)

### A) **`mode=model`** (live policy) — two processes

1. **Terminal A — policy server (Py 3.12, e.g. `conda activate lerobot-0.5.1`)**

```bash
cd src/xhum/deploy_decouple/comms
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

cd src/xhum/deploy_decouple/robot
pip install pyzmq pyyaml h5py numpy   # pyzmq required for model mode (PolicyClient)

cp config/config_zmq.example.yaml my_robot.yaml
# In YAML: mode: model
# Set policy_server_url: tcp://127.0.0.1:5555 (must match --bind)
# If your checkpoint uses observation.images.camera, set obs_camera_key: camera (see YAML comments)

python3 run.py --config ./my_robot.yaml
```

**Startup order (`mode=model` or `mode=replay`):** start **`policy_server` first**, then the ROS node. **`mode=replay_actions`** does not need the policy server.

### B1) **`mode=replay_actions`** (HDF5 actions + ROS only) — **no policy server**

Streams **actions** from `h5_path` only (**open-loop**). **Do not start `policy_server`**. No ZMQ; **pyzmq** is not required. **No RGB/depth ROS subscriptions** — you do not need live camera topics.

```bash
source /opt/ros/humble/setup.bash
cd src/xhum/deploy_decouple/robot
pip install pyyaml h5py numpy

cp config/config_zmq.example.yaml my_robot.yaml
# In YAML: mode: replay_actions  and  h5_path: /path/to/trajectory.hdf5

python3 run.py --config ./my_robot.yaml
```

### B2) **`mode=replay`** (HDF5 observations + ZMQ + ROS)

Loads **RGB + state** from the same HDF5 each step, sends them to **`policy_server`**, publishes the **returned action** (still **no ROS camera** — images come from the file). Start **`policy_server` first** (same as model). Install **pyzmq** in the ROS env. Use **`obs_camera_key`** so the image dataset key matches your file (or set **`replay_images_h5_key`** / **`replay_state_h5_key`** explicitly — see `config/config_zmq.example.yaml`).

---

## Configuration

- Copy **`robot/config/config_zmq.example.yaml`** → `my_robot.yaml` (under `robot/`).
- **Robot YAML has no `model_path`**; ZMQ-related fields are **`policy_server_url`**, etc. To change checkpoints, update **`policy_server.py --model_path`** and restart the server.
- Comments inside the YAML (Chinese) explain `mode` (`model` / `replay` / `replay_actions` / `replay_debug`), `hand_type`, **`h5_path`**, optional **`replay_*_h5_key`**, **`camera_name`** (live camera: model only), `action_rate`, optional **`obs_camera_key`**, **`image_save`**, **`joints`** (joint vector dumps + ZMQ decode check), and **`arm_command.mode`** (`cmd_pos` or `flex_freq`; arm ROS topics are fixed in `ros2_node_zmq.py`). Legacy **`replay_via_zmq`** is deprecated (see YAML header).

---

## Local checks (no robot)

From `src/xhum/deploy_decouple`:

**HDF5 → ZMQ → policy (same path as `mode=replay`, no ROS):**

1. Terminal A — policy server (Py **≥3.12**, LeRobot, same as production):

   `cd comms && python policy_server.py --model_path /path/to/pretrained_model --bind tcp://127.0.0.1:5555`

2. Copy `robot/config/config_zmq.example.yaml` to e.g. `robot/replay_debug.yaml`. Set **`mode: replay_debug`**, **`h5_path`**, **`policy_server_url: tcp://127.0.0.1:5555`**, and align **`obs_camera_key`** / optional **`replay_*_h5_key`** with your HDF5.

3. Terminal B — headless client (Py **3.10** ok; needs **pyzmq**, **h5py**, **numpy**, **pyyaml**):

   `cd robot && python3 run.py --config ./replay_debug.yaml`

For offline eval, HDF5 smoke tests, and **`compare_joints`**, see **[`scripts/README.md`](./scripts/README.md)**.

---

## Joint dumps & calibration (`joints`)

Checks that **`arm_gripper_joints`** match **after client-side multipart encode (pre-send)** vs **after server `multipart_to_obs` decode**. Files are paired by step index: `state_XXXXXXXX.npy`.

### 1) Enable dumps

1. **Robot YAML** (e.g. `robot/config/test.yaml` or your `my_robot.yaml`):

```yaml
joints:
  enabled: true
  directory: debug_client_joint   # relative to the robot process cwd, or use an absolute path
  use_timestamp_subdir: true      # false + absolute path pairs cleanly with server --joint_trace_flat
```

Writes: **`…/directory[/timestamp]/client_pre_send/state_*.npy`**.

2. **Policy server** (alongside `--save_images_dir`, etc.):

```bash
cd src/xhum/deploy_decouple/comms
python policy_server.py \
  --model_path /path/to/pretrained_model \
  --bind tcp://127.0.0.1:5555 \
  --joint_trace_dir ./debug_server_joint
```

By default a **timestamp subdir** is created under `joint_trace_dir`, then **`server_post_decode/state_*.npy`**. Add **`--joint_trace_flat`** to skip that timestamp layer when you want a fixed parent directory.

### 2) Run a short session

Same as *Local checks*: start **`policy_server`**, then **`robot/run.py`** with **`mode=replay_debug`** (or **`model`** / **`replay`**) so real ZMQ inferences occur.

**Directory alignment:** if both sides use timestamp subdirs (YAML `use_timestamp_subdir: true` and server without `--joint_trace_flat`), client and server timestamps will usually **differ**; point **`--client_dir`** and **`--server_dir`** at the **`client_pre_send`** and **`server_post_decode`** folders from **the same run**. For a single stable tree: YAML **`use_timestamp_subdir: false`** with an **absolute `directory`**, server **`--joint_trace_dir`** to the **same parent** plus **`--joint_trace_flat`**.

### 3) Run the compare script (calibration)

Commands and flags: **[`scripts/README.md`](./scripts/README.md)** (`compare_joints.py`).

---

## `replay` / `replay_actions` / `replay_debug` (details)

- **`mode=replay`:** set **`h5_path`**; requires **`policy_server`** and **pyzmq**. Observations from HDF5 each step (`replay_io/hdf5_replay_obs.py`), actions published on ROS after ZMQ inference.

- **`mode=replay_actions`:** set **`h5_path`**; no ZMQ / no **`policy_server`**. Action layout: `replay_io/hdf5_actions.py` (`puppet/joint_position` `(T, 26)` or legacy `*_align` groups).

- **`mode=replay_debug`:** no ROS; HDF5→ZMQ logging only (see *Local checks* above).

- **Compatibility:** YAML with **`mode=replay` + `replay_via_zmq:false`** is normalized to **`replay_actions`** with a deprecation warning.

---

## Stay aligned with the monolithic stack

- Robot I/O and YAML behaviour mirror **`src/xhum/deploy/ros2_deploy.py`** (including replay HDF5 loading in that file after the same update).
- **`policy/policy_agent.py`** should stay in sync with **`src/xhum/deploy/policy_agent.py`** (obs dict: `images[<short_cam>]`, `arm_gripper_joints`).

---

## Security & ops

- ZMQ TCP is **not authenticated**. Prefer **`127.0.0.1`** on one machine; on a LAN use a firewall or SSH tunnel, e.g.  
  `ssh -L 5555:127.0.0.1:5555 user@policy-host`

## Optional: systemd

For **`mode=model`** or **`mode=replay`**, run `comms/policy_server.py` under **systemd** or **supervisor** if you want auto-restart; start `run.py` after the server is listening. **`mode=replay_actions`** does not need this.
