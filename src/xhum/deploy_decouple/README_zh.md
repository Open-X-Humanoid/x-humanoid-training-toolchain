# 解耦部署（Python 3.12 策略 + Python 3.10 ROS2）

**[English](./README.md)**

LeRobot / ACT 推理依赖 **Python 3.12**（torch、`lerobot`）。Ubuntu 22.04 上 ROS2 常用 **Python 3.10**。两者塞进同一进程容易出问题，因此拆成：

| 目录 | Python | 作用 |
|------|--------|------|
| `algorithm/` | **3.12+** | `PolicyAgent` + **`policy_server.py`**，ZeroMQ **REP** 服务 |
| `ros_bridge/` | **3.10**（ROS） | 传感器/执行器 + **`PolicyClient`**，ZeroMQ **REQ** 客户端 |

报文格式见 `algorithm/policy_server.py` 文件头注释。

---

## 目录结构

```
deploy_decouple/
├── README.md                 # 英文
├── README_zh.md              # 本文件（中文）
├── algorithm/
│   ├── policy_agent.py       # ACT 封装（与 src/xhum/deployment/policy_agent.py 保持同步）
│   ├── policy_server.py      # ZMQ 策略服务进程
│   └── requirements.txt
├── ros_bridge/
│   ├── policy_client.py      # ZMQ 客户端（仅需 numpy + pyzmq）
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

## 本机部署（最常见：ROS 与策略服务同一台电脑）

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
pip install pyzmq pyyaml h5py numpy   # ROS 所用 python 缺啥装啥

cp config_zmq.example.yaml my_robot.yaml
# 确认 policy_server_url 与上面 --bind 一致，例如 tcp://127.0.0.1:5555
# 若模型里是 observation.images.camera，需在 yaml 里设 obs_camera_key: camera（见示例内注释）

python3 ros2_node_zmq.py --config ./my_robot.yaml
```

**顺序：** 一定先起 **`policy_server`**，再起 **ROS 节点**。

---

## 配置文件

- 复制 **`ros_bridge/config_zmq.example.yaml`** → **`my_robot.yaml`**。
- 示例 YAML 内为**中文注释**，说明：`mode` / `hand_type`、本机 ZMQ、`replay` 的 `h5_path`、相机话题、`action_rate`、以及 **`obs_camera_key`** 与训练时 `observation.images.<短名>` 对齐等。

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

## `mode: replay`

YAML 中设 **`mode: replay`**，并填写 **`h5_path`** 指向单个 **`trajectory.hdf5`**。此时**不会**连接 `policy_server`。

支持的 HDF5 格式见 `hdf5_actions.py`：

1. **`puppet/joint_position`**，形状 **`(T, 26)`** — 每步一条 26 维指令；
2. 旧版四组：`puppet/arm_*_align/data` 与 `end_effector_*_align/data`。

---

## 与单体部署脚本对齐

- 机器人 IO 与 YAML 行为对齐 **`src/xhum/deploy/ros2_deploy.py`**（其中 HDF5 replay 加载逻辑已与 `hdf5_actions` 一致更新）。
- **`algorithm/policy_agent.py`** 应与 **`src/xhum/deployment/policy_agent.py`** 保持同步（观测：`images[<短相机名>]` + `arm_gripper_joints`）。

---

## 安全与运维

- ZMQ **明文、无认证**。本机优先 **`127.0.0.1`**；跨机请防火墙或 SSH 隧道，例如：  
  `ssh -L 5555:127.0.0.1:5555 user@策略机`

## 可选：systemd

用 **systemd** / **supervisor** 托管 **`policy_server.py`**，崩溃自动拉起；待端口就绪后再启动 **`ros2_node_zmq.py`**。
