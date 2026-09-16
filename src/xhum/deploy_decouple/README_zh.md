# 解耦部署（Python 3.12 策略 + Python 3.10 ROS2）

**[English](./README.md)**

LeRobot / ACT 推理依赖 **Python 3.12**（torch、`lerobot`）。Ubuntu 22.04 上 ROS2 常用 **Python 3.10**。两者塞进同一进程容易出问题。本目录把「**目录 = Python 环境**」作为硬约束，而且**不需要 `pip install` 本目录**（每个入口脚本启动时自己注入 `sys.path`）：

| 目录 | Python 环境 | 作用 |
|------|-------------|------|
| `policy/` | **3.12+**（LeRobot / torch） | `policy_agent.py` + `policy_server.py`；服务进程及其依赖都在这里 |
| `robot/`  | **3.10**（ROS2 / rclpy）     | `ros2_node.py` + `policy_client.py` + `replay_debug.py` + YAML 加载 + HDF5 回放 I/O |
| `wire/`   | **两端共享**（只允许 numpy + stdlib） | `obs_codec.py`（协议）+ `trace_io.py`（PNG / 关节落盘）—— 改动需双端同步 |

**为何 ROS 侧做 Client：** 控制节拍在 ROS 进程里，每步用当前观测**主动请求**一次动作，由策略进程 **bind** 并 **应答**推理结果，更符合「闭环拉取」习惯。

报文格式见 `policy/policy_server.py` 文件头注释（实现在 `wire/obs_codec.py`）。

**模型路径：** 只在启动 **`policy/policy_server.py`** 时用 **`--model_path`**；**机端 `robot/config/*.yaml` 不要写、也不需要 `model_path`**（ZMQ 只传观测与动作，Client 不知道也不应知道 checkpoint 目录）。换模型请 **重启策略服务并更换其 `--model_path`**。

---

## 目录结构

```
deploy_decouple/
├── README.md                  # 英文
├── README_zh.md               # 本文件（中文）
├── policy/                    # Py 3.12 环境（LeRobot + torch）
│   ├── policy_agent.py        # ACT 封装（与 src/xhum/deploy/policy_agent.py 保持同步）
│   ├── policy_server.py       # ZMQ REP 服务；PolicyAgent 同目录兄弟
│   └── requirements.txt
├── robot/                     # Py 3.10 环境（ROS2 / rclpy）
│   ├── run.py                 # 入口：peek YAML `mode` 分发
│   ├── ros2_node.py           # PolicyAgentNode — model / replay / replay_actions
│   ├── replay_debug.py        # 无 rclpy 的 HDF5 → ZMQ 调试循环
│   ├── policy_client.py       # ZMQ REQ 客户端（numpy + pyzmq）
│   ├── config_loader.py       # YAML 合并 + PolicyClient 工厂（无 ROS 依赖）
│   ├── replay_io/             # HDF5 加载（开环动作 + replay/ZMQ 观测）
│   ├── config/                # 仅 `config_zmq.example.yaml` 入库，其它 YAML 已 .gitignore
│   └── requirements.txt
├── wire/                      # 两端共享；只允许 numpy + stdlib
│   ├── obs_codec.py           # 协议 meta + multipart 编解码（含 op: infer/reset）
│   └── trace_io.py            # PNG / 关节落盘 helpers（client 与 server 都用）
├── launch/                    # 示例 shell 启动脚本（见 launch/README.md）
│   ├── start_policy.example.sh
│   └── start_robot.example.sh
└── scripts/                   # 说明与脚本见 scripts/README.md
```

**本目录不需 `pip install`**。每个入口脚本（`policy/policy_server.py`、
`robot/run.py`）在启动时把同级 + `wire/` 注入 `sys.path`。依赖按环境各自在
`policy/requirements.txt` 和 `robot/requirements.txt` 管理。

---

## 更新摘要（分支 `refactor/decouple-toolchain`）

- **目录重组**：按 env 边界组织——`policy/`（Py 3.12）、`robot/`（Py 3.10）、`wire/`（共享）。原 `comms/` 目录已拆解（`policy_server.py` → `policy/`、`policy_client.py` → `robot/`、codec 与 helpers → `wire/`）。原 `robot/settings/` 拍平为 `robot/config_loader.py`。入口路径以本 README 目录树为准。
- **PolicyAgent**：若 `pretrained_model/` 下存在 `policy_preprocessor.json` 与 `policy_postprocessor.json`，推理链路与 LeRobot `predict_action` 一致（观测按训练统计量归一化，动作反归一化后再返回）；缺少上述文件时保持旧行为（直接 `select_action`，无 denorm）。
- **HDF5 评测 / 本地脚本**：见 **[`scripts/README.md`](./scripts/README.md)**（细节均在 `scripts/` 下说明）。
- **`src/xhum/deploy/policy_agent.py`**：与 `deploy_decouple/policy/policy_agent.py` 保持同步（同一套 pre/post processor 逻辑）。

---

## 本机使用：四种 **`mode`**（`model` / `replay` / `replay_actions` / `replay_debug`）

### A）**`mode=model`**（在线推理）— 两个进程

1. **终端 A — 策略服务（Python 3.12，例如 `conda activate lerobot-0.5.1`）**

```bash
cd src/xhum/deploy_decouple/policy
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

## 配置文件 `my_robot.yaml`

机端入口读的就是这份 YAML（示例文件名；也可叫别的，`--config` 指向即可）。**仓库只跟踪** `robot/config/config_zmq.example.yaml`，本地复制出来的 `my_robot.yaml` 已被 gitignore，不要提交实机路径。

```bash
cd src/xhum/deploy_decouple/robot
cp config/config_zmq.example.yaml my_robot.yaml   # 或 config/my_robot.yaml
python3 run.py --config ./my_robot.yaml
```

**机端 YAML 没有 `model_path`。** 权重只在 `policy_server.py --model_path` 加载；ZMQ 只传观测与动作。换 checkpoint：改服务端参数并**重启策略进程**。

字段级中文注释以示例 YAML 为准。下面是 README 里应先看懂的部分。

### 必填 / 常用字段

| 字段 | 作用 |
|------|------|
| **`mode`** | `model` / `replay` / `replay_actions` / `replay_debug`（见上一节） |
| **`hand_type`** | 灵巧手：`inspire` 或 `brainco`（见下一节） |
| **`robot_model`** | 机型：`tienkung2` 或 `tienkung3`（决定手臂 ROS 消息/话题；brainco 时还决定手部消息包与动作切分） |
| **`policy_server_url`** | 必须与 `policy_server.py --bind` 完全一致，例如 `tcp://127.0.0.1:5555` |
| **`h5_path`** | `replay` / `replay_actions` / `replay_debug` 必填 |
| **`camera_name`** | **仅 `mode=model`**：订阅 `/<name>/color\|depth/image_raw` |
| **`action_rate`** | 下发一拍后的 sleep 频率（Hz），默认 `20.0` |
| **`arm_command.mode`** | `cmd_pos` 或 `flex_freq`（话题名写死在 `ros2_node.py`，YAML 里不要写 topic） |

### 常可省略（会按 `hand_type` 填默认）

未写时由 `config_loader.py` 的 `HAND_TYPE_DEFAULTS` 补齐：`obs_camera_key`、`home_position`、`home_wait`、`home_hand`、`arm_spd`、`arm_cur`。

| 字段 | 说明 |
|------|------|
| **`obs_camera_key`** | 必须与 checkpoint 里 `observation.images.<短名>` 一致。inspire 默认 `camera_head`；brainco 默认 `camera` |
| **`home_position`** | `reset_home` 时 14 维手臂目标 |
| **`home_hand.left` / `right`** | 回 home 时双手位姿（单位随 `hand_type` 不同，见下节） |
| **`policy_zmq_timeout_ms`** | ZMQ 超时，默认 `120000`；`0` = 不超时 |
| **`replay_images_h5_key` / `replay_state_h5_key`** | 自动找不到 HDF5 数据集时显式指定 |
| **`image_save` / `joints`** | 落盘调试用（见后文） |

旧字段 **`replay_via_zmq`** 已弃用：若仍写 `mode=replay` 且 `replay_via_zmq: false`，加载时会当成 `replay_actions` 并打警告。

---

## 灵巧手（`hand_type`）

部署侧目前支持两种：**Inspire** 与 **BrainCo**。换手主要改 YAML 的 **`hand_type`**，并保证三件事一致：实机 ROS 接口、策略 checkpoint 的动作/观测布局、（brainco 时）**`robot_model`**。

```yaml
hand_type: inspire    # 或 brainco
robot_model: tienkung2   # 或 tienkung3
```

不支持在同一进程里混装两种手。`hand_type` 非法会在节点初始化时直接报错。

### 对照

| | **Inspire** | **BrainCo** |
|--|-------------|-------------|
| YAML | `hand_type: inspire` | `hand_type: brainco` |
| 控制话题 | `/inspire_hand/ctrl/{left,right}_hand`（`sensor_msgs/JointState`） | `/left_hand/set_motor_multi`、`/right_hand/set_motor_multi` |
| 状态话题 | `/inspire_hand/state/{left,right}_hand` | `/left_hand/motor_status`、`/right_hand/motor_status` |
| 手部位姿单位 | `[0, 1]` 浮点（四舍五入到 0.1；`1.0` ≈ 全开） | 整数 `[0, 100]`（`99` ≈ 全开） |
| 默认 `obs_camera_key` | `camera_head` | `camera` |
| 默认 `arm_spd` / `arm_cur` | `0.5` / `5.0` | `150.0` / `80.0` |
| 默认 `home_wait` | `3` | `5` |
| ROS 消息包 | 标准 `JointState` | **tienkung2**：`ros2_stark_interfaces`；**tienkung3**：`brainco_hand_msgs` |

`home_hand` 可写单个标量（广播成 6 维）或长度 6 的 list。只写 `left` 时 `right` 仍用该 `hand_type` 的默认，反之亦然。

### 26 维动作如何切成臂 + 手

策略输出长度均为 **26**（双臂 14 + 双手 12）。**切分方式**随手型和机型变化，必须与训练数据 / checkpoint 一致，否则会把手指值当关节角下发。

| `hand_type` + `robot_model` | `publish_action` 切分 |
|-----------------------------|------------------------|
| **inspire**（任意机型） | 交错：`larm(7) + lhand(6) + rarm(7) + rhand(6)` |
| **brainco + tienkung3** | 与 inspire 相同（交错） |
| **brainco + tienkung2** | 先双臂再双手：`arm(14) + lhand(6) + rhand(6)` |

在线 `mode=model` 的本体感觉 `arm_gripper_joints` 当前按交错布局组装（左臂 7 + 左手 6 + 右臂 7 + 右手 6），与 inspire 一致。若你的 brainco 模型是按 tienkung2 的「臂 14 + 手 12」训练的，请确认 checkpoint 与 `robot_model` 匹配。

### 换手 checklist

1. YAML 设对 **`hand_type`**（以及 brainco 时的 **`robot_model`**）。
2. ROS 环境能 import 上表对应的消息包；缺包会在 `_setup_*_hands` 时 `ImportError`。
3. **`obs_camera_key`** 与该 checkpoint 的视觉短名一致；不一致就在 YAML 里显式写，不要依赖默认。
4. 用与该手一起训练的 **`--model_path`**；inspire 权重不能直接套 brainco 机，反之亦然。
5. 需要改回 home 姿态时覆盖 **`home_position` / `home_hand`**，不要改代码里的默认表。

---

## 本地自检（不接真机）

在 **`src/xhum/deploy_decouple`** 下：

**用 HDF5 验证 ZMQ**（与 `mode=replay` 同链路，不启 ROS）：

1. **终端 A** — 策略服务（**Python ≥3.12**、LeRobot，与线上一致）：  
   `cd policy && python policy_server.py --model_path /path/to/pretrained_model --bind tcp://127.0.0.1:5555`

2. 复制 **`robot/config/config_zmq.example.yaml`** 为例如 **`robot/replay_debug.yaml`**，设置 **`mode: replay_debug`**、**`h5_path`**、**`policy_server_url: tcp://127.0.0.1:5555`**，并按数据对齐 **`obs_camera_key`** / 可选 **`replay_*_h5_key`**。

3. **终端 B** — 无头客户端（**Python 3.10** 即可；需 **pyzmq、h5py、numpy、pyyaml**）：  
   `cd robot && python3 run.py --config ./replay_debug.yaml`

离线评测、HDF5 冒烟、`compare_joints` 等脚本命令见 **[`scripts/README.md`](./scripts/README.md)**。

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
cd src/xhum/deploy_decouple/policy
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

命令与参数见 **[`scripts/README.md`](./scripts/README.md)**（`compare_joints.py`）。

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

**`mode=model` 或 `mode=replay`** 时建议用 **systemd** / **supervisor** 托管 **`policy/policy_server.py`**；**`mode=replay_actions`** 不需要托管策略服务。
