# Decoupled deployment (Python 3.12 policy + Python 3.10 ROS2)

**[中文说明](./README_zh.md)**

LeRobot / ACT inference needs **Python 3.12** (torch, `lerobot`). ROS2 on Ubuntu 22.04 typically uses **Python 3.10**. Mixing both in one process is fragile, so this package makes the **directory = Python environment** the design contract, and neither side has to `pip install` the deploy folder (each entry script injects its own `sys.path` at startup):

| Directory | Python env | Role |
|-----------|------------|------|
| `policy/` | **3.12+** (LeRobot / torch) | `policy_agent.py` + `policy_server.py` — the server process and everything it imports |
| `robot/`  | **3.10** (ROS2 / rclpy)    | `ros2_node.py` + `policy_client.py` + `replay_debug.py` + YAML loader + HDF5 replay I/O |
| `wire/`   | **shared** (numpy + stdlib only) | `obs_codec.py` (protocol) + `trace_io.py` (PNG / joint dumps) — both ends must agree |

**Why client on ROS:** the control loop runs in the ROS process and *pulls* one action per step from the policy process, so the ROS side is the natural **request** initiator. The policy process **binds** and **replies** with inference results.

The wire protocol is described in `policy/policy_server.py` (and implemented in `wire/obs_codec.py`).

**Model path:** only when starting **`policy/policy_server.py`** via **`--model_path`**. **Do not put `model_path` in `robot/config/*.yaml`** — the ZMQ wire carries observations and actions only; the ROS client neither sends nor needs the checkpoint directory. To switch checkpoints, **restart the policy server** with a new **`--model_path`**.

---

## Repository layout

```
deploy_decouple/
├── README.md                  # This file (English)
├── README_zh.md               # Chinese
├── policy/                    # Py 3.12 env (LeRobot + torch)
│   ├── policy_agent.py        # ACT wrapper (keep in sync with src/xhum/deploy/policy_agent.py)
│   ├── policy_server.py       # ZMQ REP server; imports PolicyAgent as a sibling
│   └── requirements.txt
├── robot/                     # Py 3.10 env (ROS2 / rclpy)
│   ├── run.py                 # Entry: peeks YAML `mode` and routes
│   ├── ros2_node.py           # PolicyAgentNode — model / replay / replay_actions
│   ├── replay_debug.py        # Headless HDF5 → ZMQ loop (no rclpy)
│   ├── policy_client.py       # ZMQ REQ client (numpy + pyzmq only)
│   ├── config_loader.py       # YAML merge + PolicyClient factory (no ROS imports)
│   ├── replay_io/             # HDF5 loaders (actions + RGB/state for ZMQ replay)
│   ├── config/                # Only `config_zmq.example.yaml` is tracked; other YAMLs are gitignored
│   └── requirements.txt
├── wire/                      # Shared between the two envs; numpy + stdlib only
│   ├── obs_codec.py           # Protocol meta + multipart encode/decode (+ op: infer/reset)
│   └── trace_io.py            # PNG / joints dump helpers (used by both client and server)
├── launch/                    # Example shell wrappers (see launch/README.md)
│   ├── start_policy.example.sh
│   └── start_robot.example.sh
└── scripts/                   # Docs + helpers; see scripts/README.md
```

**Nothing requires `pip install`**. Each entry point (`policy/policy_server.py`,
`robot/run.py`) adds the right sibling + `wire/` directories to `sys.path` at
import time. Dependencies are per-env via `policy/requirements.txt` and
`robot/requirements.txt` only.

---

## Updates (branch `refactor/decouple-toolchain`)

- **Layout:** env-boundary tree — `policy/` (Py 3.12), `robot/` (Py 3.10), `wire/` (shared). The former `comms/` directory has been dissolved (`policy_server.py` → `policy/`, `policy_client.py` → `robot/`, codec + helpers → `wire/`). The former `robot/settings/` was flattened to `robot/config_loader.py`. Use this README's tree as the source of truth for entrypoints.
- **PolicyAgent:** if `policy_preprocessor.json` and `policy_postprocessor.json` sit next to the checkpoint, inference matches LeRobot `predict_action` (normalize observations, **denormalize actions**). If those files are missing, behavior falls back to raw `select_action` (no denorm).
- **HDF5 eval / local scripts:** see **[`scripts/README.md`](./scripts/README.md)** (details live under `scripts/` only).
- **`src/xhum/deploy/policy_agent.py`** stays in sync with `deploy_decouple/policy/policy_agent.py` for the same preprocessor/postprocessor wiring.

---

## End-to-end deploy walkthrough (real robot, `mode=model`)

The usual path: **two processes on one machine**, live cameras, policy inference, ROS commands. Get this working first, then change `hand_type` / `mode`. YAML keys and hand details are later in this file.

```
Robot bring-up (arm / hand / camera drivers)     topics live
                 │
                 ▼
   ┌─────────────────────┐  ZMQ REQ/REP   ┌──────────────────────────┐
   │ policy_server.py    │ ◄────────────► │ robot/run.py             │
   │ Python 3.12         │  tcp://127.0.0.1:5555  │ Python 3.10 + ROS2      │
   │ --model_path …      │                │ --config my_robot.yaml   │
   └─────────────────────┘                └──────────────────────────┘
```

**Order:** hardware drivers → **`policy_server` first** → then **`run.py`**. If you reverse that, the client waits forever for a REP.

### 0. Prerequisites

| Item | Notes |
|------|--------|
| This repo | including the `lerobot` submodule |
| Checkpoint | training output with `pretrained_model/` (prefer `policy_preprocessor.json` / `policy_postprocessor.json` beside it) |
| Policy env | **Python 3.12+**, LeRobot + torch (e.g. `conda activate lerobot-0.5.1`) |
| ROS env | **Python 3.10**, `source /opt/ros/humble/setup.bash`; **do not** activate the lerobot conda env |
| Robot | arm, dexterous-hand, and camera drivers already running with live topics |
| Hand / body | match hardware: `inspire` or `brainco`; `tienkung2` or `tienkung3` |

### 1. One-time installs (once per env)

**Policy env:**

```bash
cd /path/to/x-humanoid-training-toolchain
# Install LeRobot the same way as training, e.g.:
#   pip install -e "lerobot/[pi]"
pip install -r src/xhum/deploy_decouple/policy/requirements.txt
```

**ROS env (lerobot conda not active):**

```bash
source /opt/ros/humble/setup.bash
# plus your robot workspace if you have one: source ~/ws/install/setup.bash
pip install -r src/xhum/deploy_decouple/robot/requirements.txt
# brainco also needs the hand msg package: tienkung2 → ros2_stark_interfaces; tienkung3 → brainco_hand_msgs
```

### 2. Write `my_robot.yaml`

```bash
cd src/xhum/deploy_decouple/robot
cp config/config_zmq.example.yaml my_robot.yaml
```

Minimum edits (example: TienKung 2 + Inspire, same machine):

```yaml
mode: model
hand_type: inspire          # brainco on a BrainCo robot
robot_model: tienkung2      # tienkung3 on TienKung 3
policy_server_url: tcp://127.0.0.1:5555
camera_name: camera         # must match /<name>/color|depth/image_raw
action_rate: 20.0
arm_command:
  mode: cmd_pos
# if the checkpoint uses observation.images.camera instead of inspire's default camera_head:
# obs_camera_key: camera
```

Do **not** put `model_path` in this YAML.

### 3. Check driver topics (separate ROS terminal)

```bash
source /opt/ros/humble/setup.bash
ros2 topic list
ros2 topic hz /camera/color/image_raw
# arm state: tienkung2 → /arm/status ; tienkung3 → /robot_state
# hand: inspire → /inspire_hand/state/left_hand ; brainco → /left_hand/motor_status
```

Without images or joint state, `run.py` sits on `Images not ready` / `Arm status not ready`.

### 4. Start both processes

**Terminal A — policy server (Python 3.12 / lerobot):**

```bash
cd src/xhum/deploy_decouple/policy
export PYTHONPATH=/path/to/x-humanoid-training-toolchain/lerobot/src:$PYTHONPATH

python policy_server.py \
  --model_path /path/to/checkpoints/last/pretrained_model \
  --bind tcp://127.0.0.1:5555
```

Wait until the model is loaded and port `5555` is bound, then start B. You can also copy `launch/start_policy.example.sh` and edit the env paths.

**Terminal B — ROS node (Python 3.10, ROS sourced):**

```bash
source /opt/ros/humble/setup.bash
cd src/xhum/deploy_decouple/robot
python3 run.py --config ./my_robot.yaml
```

Or `launch/start_robot.example.sh --config ./my_robot.yaml`.

### 5. What runs after start

In `mode=model`, `run.py` does:

1. **warm_up** (~3 s)  
2. **reset_home** — arm to `home_position`, hands to `home_hand`  
3. **reset** on the policy server (clears the ACT buffer)  
4. Loop: build `images` + `arm_gripper_joints` → ZMQ infer → split the 26-D action by `hand_type` → publish arm/hand → sleep at `action_rate`

Logs should show `hand_type=...`, `Mode=MODEL`, and `Starting remote model inference loop`.

### 6. Stop and switch checkpoints

- Stop **terminal B** first (Ctrl+C), then the policy server.  
- **New checkpoint:** change `--model_path` on A and restart A; leave YAML alone unless the vision short name changed (`obs_camera_key`).  
- **New hand / body:** edit `hand_type` / `robot_model` and use a matching checkpoint (see *Dexterous hands*).

### 7. Other setups (same binaries, different `mode`)

| Situation | What to do |
|-----------|------------|
| Laptop, no robot — smoke-test ZMQ + the model | `mode: replay_debug` + `h5_path`; still start `policy_server` first. See *Local checks* |
| Open-loop replay of a recorded trajectory | `mode: replay_actions`; **do not** start `policy_server` |
| HDF5 images through the policy, actions on the robot | `mode: replay`; start `policy_server` first |
| Policy on another host | A: `--bind tcp://0.0.0.0:5555` (or that host's IP); YAML `policy_server_url: tcp://<policy-host>:5555`; firewall or SSH tunnel |

Per-mode command details are in the next section.

---

## Same machine: four YAML **`mode`** values (`model` / `replay` / `replay_actions` / `replay_debug`)

### A) **`mode=model`** (live policy) — two processes

1. **Terminal A — policy server (Py 3.12, e.g. `conda activate lerobot-0.5.1`)**

```bash
cd src/xhum/deploy_decouple/policy
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

## Robot YAML (`my_robot.yaml`)

This is the config the ROS entry point loads (the filename is conventional; `--config` can point anywhere). The repo only tracks **`robot/config/config_zmq.example.yaml`**. Local copies such as `my_robot.yaml` are gitignored — do not commit machine-specific paths.

```bash
cd src/xhum/deploy_decouple/robot
cp config/config_zmq.example.yaml my_robot.yaml   # or config/my_robot.yaml
python3 run.py --config ./my_robot.yaml
```

**There is no `model_path` in the robot YAML.** Weights load only via `policy_server.py --model_path`. ZMQ carries observations and actions. To switch a checkpoint, change that flag and **restart the policy process**.

Field-level comments live in the example YAML (Chinese). The tables below are what you need before editing.

### Required / commonly used keys

| Key | Role |
|-----|------|
| **`mode`** | `model` / `replay` / `replay_actions` / `replay_debug` (see previous section) |
| **`hand_type`** | Dexterous hand: `inspire` or `brainco` (next section) |
| **`robot_model`** | `tienkung2` or `tienkung3` (arm ROS msgs/topics; for brainco also the hand msg package and action split) |
| **`policy_server_url`** | Must match `policy_server.py --bind` exactly, e.g. `tcp://127.0.0.1:5555` |
| **`h5_path`** | Required for `replay` / `replay_actions` / `replay_debug` |
| **`camera_name`** | **`mode=model` only**: subscribe `/<name>/color\|depth/image_raw` |
| **`action_rate`** | Sleep frequency (Hz) after each command; default `20.0` |
| **`arm_command.mode`** | `cmd_pos` or `flex_freq` (topic names are hardcoded in `ros2_node.py`; do not put topics in YAML) |

### Often omitted (filled from `hand_type` defaults)

`config_loader.py` `HAND_TYPE_DEFAULTS` supplies `obs_camera_key`, `home_position`, `home_wait`, `home_hand`, `arm_spd`, `arm_cur` when missing.

| Key | Notes |
|-----|-------|
| **`obs_camera_key`** | Must match `observation.images.<short_name>` in the checkpoint. Default `camera_head` (inspire) / `camera` (brainco) |
| **`home_position`** | 14-DoF arm target used by `reset_home` |
| **`home_hand.left` / `right`** | Hand pose at home (units depend on `hand_type`) |
| **`policy_zmq_timeout_ms`** | Default `120000`; `0` = no timeout |
| **`replay_images_h5_key` / `replay_state_h5_key`** | Explicit HDF5 paths when auto-discovery fails |
| **`image_save` / `joints`** | Optional dumps (see later sections) |

Legacy **`replay_via_zmq`**: `mode=replay` + `replay_via_zmq: false` is normalized to `replay_actions` with a deprecation warning.

---

## Dexterous hands (`hand_type`)

The deploy stack supports two hands: **Inspire** and **BrainCo**. Switching is primarily YAML **`hand_type`**, plus three invariants: the live ROS interface, the checkpoint's action/obs layout, and (for brainco) **`robot_model`**.

```yaml
hand_type: inspire    # or brainco
robot_model: tienkung2   # or tienkung3
```

You cannot mix both hands in one process. An unknown `hand_type` fails at node init.

### Comparison

| | **Inspire** | **BrainCo** |
|--|-------------|-------------|
| YAML | `hand_type: inspire` | `hand_type: brainco` |
| Command topics | `/inspire_hand/ctrl/{left,right}_hand` (`sensor_msgs/JointState`) | `/left_hand/set_motor_multi`, `/right_hand/set_motor_multi` |
| State topics | `/inspire_hand/state/{left,right}_hand` | `/left_hand/motor_status`, `/right_hand/motor_status` |
| Command units | float `[0, 1]` (rounded to 0.1; `1.0` ≈ fully open) | int `[0, 100]` (`99` ≈ fully open) |
| Default `obs_camera_key` | `camera_head` | `camera` |
| Default `arm_spd` / `arm_cur` | `0.5` / `5.0` | `150.0` / `80.0` |
| Default `home_wait` | `3` | `5` |
| ROS msg package | standard `JointState` | **tienkung2**: `ros2_stark_interfaces`; **tienkung3**: `brainco_hand_msgs` |

`home_hand` accepts a scalar (broadcast to 6) or a length-6 list. Setting only `left` keeps the `hand_type` default for `right`, and vice versa.

### How the 26-D action is split

Policy output is always **26-D** (14 arm + 12 hands). **The split** depends on hand and robot model and must match training / the checkpoint — otherwise finger values are published as arm joints.

| `hand_type` + `robot_model` | `publish_action` split |
|-----------------------------|------------------------|
| **inspire** (any model) | interleaved: `larm(7) + lhand(6) + rarm(7) + rhand(6)` |
| **brainco + tienkung3** | same interleaved layout as inspire |
| **brainco + tienkung2** | contiguous arms first: `arm(14) + lhand(6) + rhand(6)` |

Live `mode=model` proprioception (`arm_gripper_joints`) is currently assembled interleaved (left arm 7 + left hand 6 + right arm 7 + right hand 6), same as inspire. If a brainco checkpoint was trained with the tienkung2 `arm(14)+hands(12)` layout, keep `robot_model` aligned with that checkpoint.

### Switching hands

1. Set **`hand_type`** (and **`robot_model`** when using brainco).
2. The ROS env must import the message package in the table above; missing packages raise `ImportError` in `_setup_*_hands`.
3. Set **`obs_camera_key`** to the checkpoint's vision short name; do not rely on the hand default if they differ.
4. Point **`--model_path`** at a checkpoint trained for that hand. Inspire weights are not drop-in on a brainco robot, and vice versa.
5. Override **`home_position` / `home_hand`** in YAML if you need a different home; do not edit the defaults table in code.

---

## Local checks (no robot)

From `src/xhum/deploy_decouple`:

**HDF5 → ZMQ → policy (same path as `mode=replay`, no ROS):**

1. Terminal A — policy server (Py **≥3.12**, LeRobot, same as production):

   `cd policy && python policy_server.py --model_path /path/to/pretrained_model --bind tcp://127.0.0.1:5555`

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
cd src/xhum/deploy_decouple/policy
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

For **`mode=model`** or **`mode=replay`**, run `policy/policy_server.py` under **systemd** or **supervisor** if you want auto-restart; start `run.py` after the server is listening. **`mode=replay_actions`** does not need this.
