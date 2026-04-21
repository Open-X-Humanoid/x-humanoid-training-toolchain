# 解耦部署（Python 3.12 策略 + Python 3.10 ROS2）

**[English](./README.md)**

LeRobot / ACT 推理依赖 **Python 3.12**（torch、`lerobot`）。Ubuntu 22.04 上 ROS2 常用 **Python 3.10**。两者塞进同一进程容易出问题，因此拆成：

| 目录 | Python | 作用 |
|------|--------|------|
| `policy/` | **3.12+** | 仅 **`PolicyAgent`**（LeRobot / ACT）；由策略服务进程导入 |
| `comms/` | **3.12+**（服务）/ **3.10**（客户端） | **`policy_server.py`**、**`policy_client.py`**、**`zmq_obs_codec.py`**、**`utils.py`** — ZMQ 报文与进程 |
| `robot/` | **3.10**（ROS） | 机器人与 HDF5；**`mode=model`** 或 **`mode=replay`** 时从 `comms/` 加载 **`PolicyClient`** |

**为何 ROS 侧做 Client：** 控制节拍在 ROS 进程里，每步用当前观测**主动请求**一次动作，由策略进程 **bind** 并 **应答**推理结果，更符合「闭环拉取」习惯。

报文格式见 `comms/policy_server.py` 文件头注释。

**模型路径：** 只在启动 **`comms/policy_server.py`** 时用 **`--model_path`**；**机端 `robot/*.yaml` 不要写、也不需要 `model_path`**（ZMQ 只传观测与动作，Client 不知道也不应知道 checkpoint 目录）。换模型请 **重启策略服务并更换其 `--model_path`**。

---

## 目录结构

```
deploy_decouple/
├── README.md                 # 英文
├── README_zh.md              # 本文件（中文）
├── policy/
│   ├── policy_agent.py       # ACT 封装（与 src/xhum/deploy/policy_agent.py 保持同步）
│   └── requirements.txt
├── comms/
│   ├── policy_server.py      # ZMQ REP 策略服务（Py3.12 + LeRobot；导入 PolicyAgent）
│   ├── policy_client.py      # ZMQ REQ 客户端（numpy + pyzmq）；model / replay / replay_debug
│   ├── zmq_obs_codec.py      # 观测 multipart 编解码
│   └── utils.py              # 存图 / YAML 整型等共用工具
├── robot/
│   ├── replay_io/            # HDF5 加载（开环动作 + replay/ZMQ 观测）
│   ├── settings/             # YAML 合并 + PolicyClient 构造（无 ROS）
│   ├── config/               # 示例 YAML（复制到 robot/ 根目录或写绝对路径）
│   ├── ros2_node_zmq.py      # 节点实现（也可直接运行）
│   ├── run.py                # 薄入口 — model / replay / replay_actions（ROS2）或 replay_debug（无 ROS）
│   └── requirements.txt
├── launch/                   # 示例 shell 启动脚本（见 launch/README.md）
│   ├── start_policy.example.sh
│   └── start_robot.example.sh
└── scripts/                  # 本地脚本目录（仓库 .gitignore，不进版本库；自管评测/冒烟等）
```

---

## 更新摘要（分支 `refactor/decouple-toolchain`）

- **目录重组**：原 `algorithm/`、`ros_bridge/` 等迁入 `policy/`、`comms/`、`robot/`、`launch/` 等；ZMQ 与 ROS 入口路径以本 README 目录树为准。
- **PolicyAgent**：若 `pretrained_model/` 下存在 `policy_preprocessor.json` 与 `policy_postprocessor.json`，推理链路与 LeRobot `predict_action` 一致（观测按训练统计量归一化，动作反归一化后再返回）；缺少上述文件时保持旧行为（直接 `select_action`，无 denorm）。
- **HDF5 评测**：可在本机 `scripts/` 下放评测脚本（该目录已 .gitignore）；典型用法为从 HDF5 构造观测、逐步推理，与 `--gt_key` 下一行 GT 对比，并按关节维度统计 `diff(pred - gt_next)`。
- **`src/xhum/deploy/policy_agent.py`**：与 `deploy_decouple/policy/policy_agent.py` 保持同步（同一套 pre/post processor 逻辑）。

---

## 本机使用：四种 **`mode`**（`model` / `replay` / `replay_actions` / `replay_debug`）

### A）**`mode=model`**（在线推理）— 两个进程

1. **终端 A — 策略服务（Python 3.12，例如 `conda activate lerobot-0.5.1`）**

```bash
cd src/xhum/deploy_decouple/comms
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

cd src/xhum/deploy_decouple/robot
pip install pyzmq pyyaml h5py numpy   # model 模式需要 pyzmq（PolicyClient）

cp config/config_zmq.example.yaml my_robot.yaml
# YAML 里 mode: model
# policy_server_url 与上面 --bind 一致，例如 tcp://127.0.0.1:5555
# 若模型里是 observation.images.camera，需在 yaml 里设 obs_camera_key: camera（见示例内注释）

python3 run.py --config ./my_robot.yaml
```

**启动顺序：** **`mode=model`** 或 **`mode=replay`** 时，先起 **`policy_server`**，再起 **ROS 节点**；**`mode=replay_actions`** 不需要策略服务。

### B1）**`mode=replay_actions`**（仅 HDF5 动作 + ROS）— **不需要策略服务**

只按 `h5_path` **纯开环**回放动作，**不要启动 `policy_server`**。无 ZMQ；可不装 **pyzmq**；**不**订阅 ROS 相机。

```bash
source /opt/ros/humble/setup.bash
cd src/xhum/deploy_decouple/robot
pip install pyyaml h5py numpy

cp config/config_zmq.example.yaml my_robot.yaml
# YAML：mode: replay_actions ，h5_path: .../trajectory.hdf5

python3 run.py --config ./my_robot.yaml
```

### B2）**`mode=replay`**（HDF5 观测 + ZMQ + ROS）

每步从 HDF5 读 **RGB + 状态**，经 **`PolicyClient`** 发给 **`policy_server`**，再下发**返回的动作**（仍**不**订阅 ROS 相机）。需先起 **`policy_server`**，ROS 环境安装 **pyzmq**。请配置 **`obs_camera_key`** 与 HDF5 中图像路径一致，或显式设置 **`replay_images_h5_key`** / **`replay_state_h5_key`**（见 `config/config_zmq.example.yaml`）。

---

## 配置文件

- 复制 **`robot/config/config_zmq.example.yaml`** → **`my_robot.yaml`**（放在 `robot/` 下）。
- **机端 YAML 不含 `model_path`**；与 ZMQ 相关的只有 **`policy_server_url`** 等。换模型只改 **`policy_server.py --model_path`** 并重启服务。
- 示例 YAML 内为**中文注释**，说明：`mode`（`model` / `replay` / `replay_actions` / `replay_debug`）、`hand_type`、**`h5_path`**、可选 **`replay_*_h5_key`**、**`camera_name`**（**仅 model** 订阅实时相机）、`action_rate`、**`obs_camera_key`**、**`image_save`**、**`joints`**（关节向量落盘与 ZMQ 编解码校验）、**`arm_command.mode`**（`cmd_pos` 或 `flex_freq`；手臂话题名写死在 `ros2_node_zmq.py`）等；旧字段 **`replay_via_zmq`** 已弃用（见示例 YAML 顶栏）。

---

## 本地自检（不接真机）

在 **`src/xhum/deploy_decouple`** 下：

**说明：** 仓库 **`.gitignore`** 已忽略 **`scripts/`**，下列命令中的脚本需在本机自行创建该目录并放入对应 `.py`（路径均相对 `src/xhum/deploy_decouple`）。

**用 HDF5 验证 ZMQ**（与 `mode=replay` 同链路，不启 ROS）：

1. **终端 A** — 策略服务（**Python ≥3.12**、LeRobot，与线上一致）：  
   `cd comms && python policy_server.py --model_path /path/to/pretrained_model --bind tcp://127.0.0.1:5555`

2. 复制 **`robot/config/config_zmq.example.yaml`** 为例如 **`robot/replay_debug.yaml`**，设置 **`mode: replay_debug`**、**`h5_path`**、**`policy_server_url: tcp://127.0.0.1:5555`**，并按数据对齐 **`obs_camera_key`** / 可选 **`replay_*_h5_key`**。

3. **终端 B** — 无头客户端（**Python 3.10** 即可；需 **pyzmq、h5py、numpy、pyyaml**）：  
   `cd robot && python3 run.py --config ./replay_debug.yaml`

| 命令 | 作用 |
|------|------|
| `python scripts/test_replay_hdf5.py /path/to/trajectory.hdf5` | 只测 HDF5 **动作**加载（与 **`mode=replay_actions`** 一致）；**无 ZMQ**。 |
| `python scripts/test_policy_agent_fake.py --model_path .../pretrained_model` | 进程内 **`PolicyAgent`** + 随机观测测一帧（**无 ZMQ**，需 **≥3.12**）。 |
| `python scripts/eval_policy_from_hdf5.py --h5_path ... --model_path ...` | 逐帧 **`PolicyAgent.inference`**，打印 **`pred`**、**`gt_next`**（**`--gt_key`** 下一行）、**`diff(pred-gt_next)`**。默认输入：**`observations/rgb_images/camera_camera`** + **`puppet/joint_position`**，**`obs_camera_key=camera`**；默认 GT：**`master/joint_position`**。均可 CLI 覆盖。**`--quiet`** 关闭逐步打印。需 **≥3.12** + LeRobot + **h5py**。 |

**HDF5 离线推理：** 在 **`src/xhum/deploy_decouple`** 下设置 **`PYTHONPATH`**（含 **`lerobot/src`**），例如：

```bash
export PYTHONPATH=/你的路径/x-humanoid-training-toolchain/lerobot/src:$PYTHONPATH
python scripts/eval_policy_from_hdf5.py \
  --h5_path /path/to/trajectory.hdf5 \
  --model_path /path/to/pretrained_model \
  --max_steps 100
```

可选 **`--gt_key`**、**`--replay_images_h5_key`**、**`--replay_state_h5_key`**、**`--obs_camera_key`**（须与 checkpoint 视觉短名一致）、**`--start`**。**`--max_steps 0`** 跑满。**`--quiet`** 不打印每步向量。

---

## 关节向量落盘与校准（`joints`）

用于核对 **`arm_gripper_joints`** 在客户端 **multipart 编码后、ZMQ 发送前** 与策略服务 **`multipart_to_obs` 解码后** 是否一致（步号 `state_XXXXXXXX.npy` 一一对应）。

### 1）打开落盘

1. **机端 YAML**（如 `robot/config/test.yaml` 或自用的 `my_robot.yaml`）：

```yaml
joints:
  enabled: true
  directory: debug_client_joint    # 相对 robot 进程 cwd，或写绝对路径
  use_timestamp_subdir: true       # false + 绝对路径 便于与服务端固定父目录对齐
```

写出：**`…/directory[/时间戳]/client_pre_send/state_*.npy`**。

2. **策略服务**（与 `--save_images_dir` 等并列）：

```bash
cd src/xhum/deploy_decouple/comms
python policy_server.py \
  --model_path /path/to/pretrained_model \
  --bind tcp://127.0.0.1:5555 \
  --joint_trace_dir ./debug_server_joint
```

默认在 `joint_trace_dir` 下再建**一层时间戳**，再写 **`server_post_decode/state_*.npy`**。若不要这层时间戳（与服务端/客户端固定在同一父目录下对比），加 **`--joint_trace_flat`**。

### 2）跑一段联调

与「本地自检」相同：先起 **`policy_server`**，再起 **`mode=replay_debug`**（或 **`model` / `replay`**）的 **`robot/run.py`**，让若干步 ZMQ 推理实际发生。

**目录对齐说明：** 客户端、服务端若都 **`use_timestamp_subdir: true`**（且服务端未加 `--joint_trace_flat`），会得到**两个不同时间戳**子目录；对比时 `--client_dir` / `--server_dir` 要分别指向**同一次联调**里实际生成的 `client_pre_send` 与 `server_post_decode` 路径。单机固定目录：YAML 里 **`use_timestamp_subdir: false`** 且 **`directory` 用绝对路径**；服务端 **`--joint_trace_dir` 指向同一父路径** 并加 **`--joint_trace_flat`**，则客户端、服务端子目录在同一父路径下，便于脚本一次写死。

### 3）执行对比脚本（校准）

在 **`src/xhum/deploy_decouple`** 下（需 **numpy**）：

```bash
python3 scripts/compare_joints.py \
  --client_dir /path/to/.../client_pre_send \
  --server_dir /path/to/.../server_post_decode
```

可选：**`--atol`**、**`--rtol`**（默认约 `1e-5`）、**`--verbose`**。终端打印 **`PASS`/`FAIL`**；**退出码 0** 表示全部序号对齐且 **`np.allclose`** 通过，**非 0** 表示缺文件、shape 不一致或数值差异超阈值。

---

## `replay` / `replay_actions` / `replay_debug`（补充）

- **`mode=replay`**：填写 **`h5_path`**，需 **`policy_server`** 与 **pyzmq**；每步观测从 HDF5 读取（**`replay_io/hdf5_replay_obs.py`**），经 ZMQ 取动作后在 ROS 上发布。手臂/手话题仍用于 `reset_home` / `reach_target_joint`。

- **`mode=replay_actions`**：填写 **`h5_path`**；**无 ZMQ**、无 **`policy_server`**。动作 HDF5 格式见 **`replay_io/hdf5_actions.py`**（`puppet/joint_position` `(T,26)` 或旧版 `*_align`）。

- **`mode=replay_debug`**：无 ROS；仅 HDF5→ZMQ 日志（见上文「本地自检」）。

- **兼容**：仍写 **`mode=replay` + `replay_via_zmq:false`** 时，加载时等价于 **`replay_actions`** 并打印弃用警告。

---

## 与单体部署脚本对齐

- 机器人 IO 与 YAML 行为对齐 **`src/xhum/deploy/ros2_deploy.py`**（其中 HDF5 replay 加载逻辑已与 `replay_io.hdf5_actions` 一致更新）。
- **`policy/policy_agent.py`** 应与 **`src/xhum/deploy/policy_agent.py`** 保持同步（观测：`images[<短相机名>]` + `arm_gripper_joints`）。

---

## 安全与运维

- ZMQ **明文、无认证**。本机优先 **`127.0.0.1`**；跨机请防火墙或 SSH 隧道，例如：  
  `ssh -L 5555:127.0.0.1:5555 user@策略机`

## 可选：systemd

**`mode=model` 或 `mode=replay`** 时建议用 **systemd** / **supervisor** 托管 **`comms/policy_server.py`**；**`mode=replay_actions`** 不需要托管策略服务。
