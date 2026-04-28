
# x-humanoid training toolchain (xhum)

[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Project Page](https://img.shields.io/badge/Project%20Page-RoboMIND-blue.svg)](https://x-humanoid-robomind.github.io/)
[![arXiv](https://badgen.net/badge/icon/arXiv?icon=awesome&label&color=red&style=flat-square)](https://arxiv.org/abs/2412.13877)
[![Dataset](https://img.shields.io/badge/Dataset-flopsera-000000.svg)](http://open.flopsera.com/flopsera-open/data-details/RoboMIND)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-RoboMIND-000000.svg)](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND)

**[English](./README.md)｜简体中文**

## 项目介绍

本项目是 RoboMIND 数据集和天工机器人的训练与部署工具链。基于 [LeRobot](https://github.com/huggingface/lerobot) 开源框架（以 git submodule 方式集成），自定义工具链代码（`xhum`）与上游代码完全解耦。

- 支持开源多本体数据集 RoboMIND
- 兼容 LeRobotDataset V3
- HDF5 到 LeRobot 数据集转换管线
- 天工机器人 ROS2 部署（BrainCo / Inspire 灵巧手）
- 具身操作训练流程（ACT、Diffusion Policy）

<table>
    <tbody>
    <tr><th>模块</th><th>描述</th></tr>
    <tr>
       <td align="center"><a href="https://github.com/x-humanoid-robomind/x-humanoid-robomind.github.io">RoboMIND 数据集</a></td>
       <td>汇集了多种机器人平台的操作数据，包含 479 种任务、96 类物体的 10.7 万条真实世界演示轨迹。</td>
    </tr>
    <tr>
       <td align="center"><a href="https://github.com/x-humanoid-robomind/TienKung_URDF">天工 URDF</a></td>
       <td>包含完整的机器人描述文件（URDF）和网格文件（STL），支持 ROS/Gazebo 仿真。</td>
    </tr>
    <tr>
       <td align="center"><a href="https://github.com/x-humanoid-robomind/TienKung_ROS">天工软件系统</a></td>
       <td>基于 ROS 框架开发的底层硬件控制系统，包含本体控制、遥控器通信等模块。</td>
    </tr>
    <tr>
       <td align="center"><a href="https://github.com/x-humanoid-robomind/TienKung_Docs">天工文档</a></td>
       <td>天工机器人用户手册和 SDK 文档，涵盖 Lite 和 Pro 版本。</td>
    </tr>
    </tbody>
</table>

## 项目结构

```
x-humanoid-training-toolchain/
├── lerobot/                        # Git submodule → huggingface/lerobot (v0.5.1)
├── src/xhum/                       # 自定义工具链（与 lerobot 完全解耦）
│   ├── convert/
│   │   ├── hdf5_to_lerobot.py      # HDF5 → LeRobot V3 数据集转换
│   │   ├── convert.example.sh      # 复制为 convert.sh 后编辑
│   │   └── configs/                # 数据集转换 config（HDF5 key → feature 映射）
│   ├── train/
│   │   ├── run_train_native.example.sh  # 单数据集 lerobot-train 包装脚本
│   │   ├── train_multi.py          # 多数据集训练入口
│   │   └── configs/                # 多数据集训练配置示例
│   ├── deploy/                      # 统一 ROS2 实机部署（./scripts/xhum-run xhum.deploy.ros2_deploy）
│   │   ├── policy_agent.py
│   │   ├── ros2_deploy.py
│   │   └── config.yaml
│   └── deploy_decouple/             # Py3.12 策略服务 + Py3.10 ROS ZMQ（见该目录 README）
├── scripts/
│   └── xhum-run                     # 不设 PYTHONPATH 也可运行：./scripts/xhum-run xhum.<模块> …
├── pyproject.toml                   # Python 包元数据（本仓库通过 scripts/xhum-run 使用 xhum）
├── Makefile                         # 快速安装命令
└── static/                          # 演示素材
```

## 环境要求

- Python >= 3.12
- Git
- （可选）ROS2 Humble/Iron，用于实机部署
- （可选）支持 CUDA 的 GPU，用于模型训练

## 安装

### 1. 克隆仓库（含子模块）

```bash
git clone --recurse-submodules https://github.com/Open-X-Humanoid/x-humanoid-training-toolchain.git
cd x-humanoid-training-toolchain
```

如果已经克隆但没带 `--recurse-submodules`：

```bash
git submodule update --init --recursive
```

### 2. 一键安装（默认只装 LeRobot）

```bash
make install
```

当前默认**只**执行：

```bash
pip install -e ./lerobot    # 从子模块安装 LeRobot
```

`src/xhum` 工具链**默认不** `pip install`；在仓库根目录用 **`./scripts/xhum-run`** 即可直接跑各模块（已自动设置 `PYTHONPATH=src`）。

```bash
./scripts/xhum-run xhum.convert.hdf5_to_lerobot --help
./scripts/xhum-run xhum.deploy.ros2_deploy --config src/xhum/deploy/config.yaml
```

等价写法：`PYTHONPATH=src python -m xhum....`

`make install` / `make install-all` **只装 LeRobot**；xhum 一律用 **`./scripts/xhum-run`**（或 `PYTHONPATH=src`）。

### 3. （可选）安装开发工具

```bash
make install-dev    # 仅 LeRobot + pre-commit / pytest / ruff（不安装 xhum 包）
```

## 更新 LeRobot

更新 LeRobot 子模块到最新上游版本：

```bash
make update-lerobot
make install-lerobot
```

或手动操作：

```bash
cd lerobot
git fetch origin
git checkout main
git pull origin main
cd ..
pip install -e ./lerobot
```

锁定到特定版本（如 v0.5.1）：

```bash
cd lerobot
git fetch --tags
git checkout v0.5.1
cd ..
pip install -e ./lerobot
```

## 使用说明

### 数据集转换

将 HDF5 格式的 RoboMIND 数据转换为 LeRobot V3 数据集格式。

推荐（在**仓库根目录**执行）：

```bash
./scripts/xhum-run xhum.convert.hdf5_to_lerobot --help

./scripts/xhum-run xhum.convert.hdf5_to_lerobot \
  --config src/xhum/convert/configs/Tien_Kung_Gello_1RGB.json \
  --repo_id my_dataset \
  --src_root /path/to/hdf5/data \
  --tgt_path /path/to/output \
  --task_name pick_cup
```


| 参数 | 说明 |
|---|---|
| `--config` | 数据集 config JSON 路径（含 `dataset.fps`、`dataset.robot_type` 与 HDF5 → feature 映射） |
| `--repo_id` | 数据集标识符（输出子目录名） |
| `--src_root` | 包含 HDF5 episode 文件夹的源目录 |
| `--tgt_path` | 转换后数据集的输出父目录 |
| `--task_name` | 任务名称（默认 `default_task`） |

> `fps` / `robot_type` 不是 CLI 参数；写在 config JSON 的 `dataset` 段里。详见 [`src/xhum/convert/README.md`](src/xhum/convert/README.md)。

### 模型训练

#### 单数据集（`lerobot-train`）

复制示例脚本，按需修改后运行：

```bash
cp src/xhum/train/run_train_native.example.sh src/xhum/train/run_train_native.sh
# 编辑 dataset.repo_id、output_dir、HF_LEROBOT_HOME 等
bash src/xhum/train/run_train_native.sh
```

脚本通过 CLI flags 直接调 `lerobot-train`（`--dataset.repo_id`、`--policy.type=act`、`--output_dir`、`--steps` 等），追加参数会原样传过去：`bash run_train_native.sh --resume=true`。

#### 多数据集（`xhum.train.train_multi`）

合并多个 LeRobot V3 数据集联合训练（自动按公共特征求交集 + 跨集 stats 聚合）：

```bash
./scripts/xhum-run xhum.train.train_multi --config src/xhum/train/configs/multi_train_example.json
```

config schema 与限制（action/state 维度必须一致）见 [`src/xhum/train/README.md`](src/xhum/train/README.md)。

### 可视化

```bash
lerobot-dataset-viz --repo-id my_dataset --episode-index 0 --root /path/to/dataset
```

<div style="display: flex;">
  <img src="./static/demo1.gif" width="300">
  <img src="./static/demo2.gif" width="300">
</div>

<div style="display: flex;">
  <img src="./static/demo3.gif" width="300">
  <img src="./static/demo4.gif" width="300">
</div>

### ROS2 部署

仓库根目录直接运行（`hand_type` 在 `config.yaml` 里选 `brainco` / `inspire`）：

```bash
./scripts/xhum-run xhum.deploy.ros2_deploy --config /path/to/src/xhum/deploy/config.yaml
# 编辑 config.yaml：model_path、h5_path（replay）、hand_type、mode 等
```

解耦部署（策略 Python3.12 + ROS Python3.10）见 **`src/xhum/deploy_decouple/README_zh.md`**。

动作向量布局（26 维）：
- `[0:7]` 左臂，`[7:13]` 左手，`[13:20]` 右臂，`[20:26]` 右手

## 计划

- 集成更多前沿机器人算法
- 支持天工系列的具身操作能力

## 致谢

基于 Hugging Face 的 [LeRobot](https://github.com/huggingface/lerobot) 框架构建，感谢！

## 讨论

如果您对 RoboMIND 有兴趣，欢迎加入我们的社群进行讨论。

<img src="./static/qrcode.png" border=0 width=30%>
