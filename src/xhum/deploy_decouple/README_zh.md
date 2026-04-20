# 解耦部署（Python 3.12 策略 + Python 3.10 ROS2）

**[English](./README.md)**

LeRobot / ACT 推理依赖 **Python 3.12**（torch、`lerobot`）。Ubuntu 22.04 上 ROS2 常用 **Python 3.10**。两者塞进同一进程容易出问题，因此拆成：

| 目录 | Python | 作用 |
|------|--------|------|
| `algorithm/` | **3.12+** | `PolicyAgent` + **`policy_server.py`**，ZeroMQ **REP**（服务端） |
| `ros_bridge/` | **3.10**（ROS） | 传感器/执行器；**`mode=model`** 时用 **`PolicyClient`**，ZeroMQ **REQ**（客户端） |

**为何 ROS 侧做 Client：** 控制节拍在 ROS 进程里，每步用当前观测**主动请求**一次动作，由策略进程 **bind** 并 **应答**推理结果，更符合「闭环拉取」习惯。

报文格式见 `algorithm/policy_server.py` 文件头注释。

---

## 目录结构

```
deploy_decouple/
├── README.md                 # 英文
├── README_zh.md              # 本文件（中文）
├── algorithm/
│   ├── policy_agent.py       # ACT 封装（与 src/xhum/deploy/policy_agent.py 保持同步）
│   ├── policy_server.py      # ZMQ 策略服务进程
│   └── requirements.txt
├── ros_bridge/
│   ├── policy_client.py      # ZMQ 客户端（numpy + pyzmq）；仅 mode=model 时才会被 import
│   ├── hdf5_actions.py       # replay 用 HDF5 读取（joint_position 或旧版 *_align）
│   ├── ros2_node_zmq.py      # ROS2 节点（model / replay）
│   ├── config_zmq.example.yaml   # 带详细注释的示例（复制为 my_robot.yaml）
│   └── requirements.txt
└── scripts/
    ├── run_local_checks.py   # protocol | e2e 自检
    ├── test_replay_hdf5.py   # 离线验证 replay 用 HDF5 能否加载
    └── test_policy_agent_fake.py  # 加载 PolicyAgent + 随机观测（无 ROS）
```

---

## 本机使用：按 **`mode=model`** 或 **`mode=replay`** 选择流程

### A）**`mode=model`**（在线推理）— 两个进程

1. **终端 A — 策略服务（Python 3.12，例如 `conda activate lerobot-0.5.1`）**

```bash
cd src/xhum/deploy_decouple/algorithm
export PYTHONPATH=/你的路径/x-humanoid-training-toolchain/lerobot/src:$PYTHONPATH
pip install pyzmq   # 本环境装一次即可

python policy_server.py \
  --model_path /你的路径/checkpoints/last/pretrained_model \
  --bind tcp://127.0.0.1:5555
```

建议 **`127.0.0.1`**：只本机可连，比 `0.0.0.0` 更省事、略安全。

2. **终端 B — ROS（不要 `conda activate lerobot`，避免和 ROS 的 Python 3.10 混用）**

```bash
source /opt/ros/humble/setup.bash
# 如有 workspace 再 source install/setup.bash

cd src/xhum/deploy_decouple/ros_bridge
pip install pyzmq pyyaml h5py numpy   # model 模式需要 pyzmq（PolicyClient）

cp config_zmq.example.yaml my_robot.yaml
# YAML 里 mode: model
# policy_server_url 与上面 --bind 一致，例如 tcp://127.0.0.1:5555
# 若模型里是 observation.images.camera，需在 yaml 里设 obs_camera_key: camera（见示例内注释）

python3 ros2_node_zmq.py --config ./my_robot.yaml
```

**启动顺序（仅 model）：** 先起 **`policy_server`**，再起 **ROS 节点**。

### B）**`mode=replay`**（HDF5 回放）— **不需要策略服务**

只按 `h5_path` **纯开环**回放，**不要启动 `policy_server`**。ROS 节点**不会**建立 ZMQ 连接；**不会**加载 `policy_client` / **pyzmq**；**不会**订阅 RGB/深度（无需相机话题与 `camera_name`）。

```bash
source /opt/ros/humble/setup.bash
cd src/xhum/deploy_decouple/ros_bridge
pip install pyyaml h5py numpy   # 仅 replay 时可不装 pyzmq

cp config_zmq.example.yaml my_robot.yaml
# YAML：mode: replay ，并填写 h5_path 指向 trajectory.hdf5
# replay 下 policy_server_url、camera_name 均不参与订阅（无相机）

python3 ros2_node_zmq.py --config ./my_robot.yaml
```

---

## 配置文件

- 复制 **`ros_bridge/config_zmq.example.yaml`** → **`my_robot.yaml`**。
- 示例 YAML 内为**中文注释**，说明：`mode` / `hand_type`、本机 ZMQ（**仅 model**）、`replay` 的 **`h5_path`**、**`camera_name`**（**仅 model**；replay 不订阅相机）、`action_rate`、**`obs_camera_key`** 与训练时 `observation.images.<短名>` 对齐，以及 **`arm_command.mode`**（`cmd_pos` 或 `flex_freq`；手臂话题名写死在 `ros2_node_zmq.py`）等。

---

## 本地自检（不接真机 / 可不启 ROS）

在目录 **`src/xhum/deploy_decouple`** 下执行：

| 命令 | 依赖 |
|------|------|
| `python scripts/run_local_checks.py protocol` | `numpy`、`pyzmq` — 模拟 ZMQ 往返 |
| `python scripts/run_local_checks.py e2e --model_path .../pretrained_model` | **Python ≥3.12**、LeRobot、torch、pyzmq；脚本会给子进程加 `lerobot/src`；可用环境变量 **`LEROBOT_PYTHON`** 指定解释器，否则尝试 `python3.12` 或 `/opt/conda/envs/lerobot-0.5.1/bin/python` |
| `python scripts/test_replay_hdf5.py /path/to/trajectory.hdf5` | `h5py`、`numpy` — 与 **`mode=replay`** 相同的 HDF5 读取逻辑 |
| `python scripts/test_policy_agent_fake.py --model_path .../pretrained_model` | **Python ≥3.12**、LeRobot、torch、opencv — 直接测 **`algorithm/policy_agent.py`** + 随机观测 |

---

## `mode: replay`（补充说明）

YAML 中设 **`mode: replay`**，并填写 **`h5_path`**。无需 **`policy_server`**、无需 **pyzmq**、**不建相机订阅**（见上文 **§B**）。手臂/手的话题仍用于 `reset_home` / `reach_target_joint` 等与真机交互。

支持的 HDF5 格式见 `hdf5_actions.py`：

1. **`puppet/joint_position`**，形状 **`(T, 26)`** — 每步一条 26 维指令；
2. 旧版四组：`puppet/arm_*_align/data` 与 `end_effector_*_align/data`。

---

## 与单体部署脚本对齐

- 机器人 IO 与 YAML 行为对齐 **`src/xhum/deploy/ros2_deploy.py`**（其中 HDF5 replay 加载逻辑已与 `hdf5_actions` 一致更新）。
- **`algorithm/policy_agent.py`** 应与 **`src/xhum/deploy/policy_agent.py`** 保持同步（观测：`images[<短相机名>]` + `arm_gripper_joints`）。

---

## 安全与运维

- ZMQ **明文、无认证**。本机优先 **`127.0.0.1`**；跨机请防火墙或 SSH 隧道，例如：  
  `ssh -L 5555:127.0.0.1:5555 user@策略机`

## 可选：systemd

**仅 `mode=model` 时**建议用 **systemd** / **supervisor** 托管 **`policy_server.py`**，崩溃自动拉起；待端口就绪后再启动 **`ros2_node_zmq.py`**。纯 replay 不需要托管策略服务。
