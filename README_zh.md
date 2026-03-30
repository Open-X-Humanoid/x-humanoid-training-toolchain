
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
│   │   ├── convert.sh              # 转换示例脚本
│   │   └── configs/                # 数据集 schema 与训练配置
│   └── deployment/
│       ├── policy_agent.py          # 策略推理封装
│       ├── ros2_brainco.py          # BrainCo 灵巧手 ROS2 部署
│       └── ros2_inspire.py          # Inspire 灵巧手 ROS2 部署
├── pyproject.toml                   # xhum 包配置
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

### 2. 一键安装

```bash
make install
```

内部执行两步：

```bash
pip install -e ./lerobot    # 从子模块安装 LeRobot
pip install -e .            # 安装 xhum 工具链
```

### 3. （可选）安装开发工具

```bash
make install-dev
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

将 HDF5 格式的 RoboMIND 数据转换为 LeRobot V3 数据集格式：

```bash
xhum-convert \
  --config src/xhum/convert/configs/Tien_Kung_Gello_1RGB.json \
  --repo_id my_dataset \
  --src_root /path/to/hdf5/data \
  --tgt_path /path/to/output \
  --task_name pick_cup \
  --fps 30 \
  --robot_type tienkung
```

| 参数 | 说明 |
|---|---|
| `--config` | 数据集特征 schema JSON 文件路径 |
| `--repo_id` | 数据集标识符 |
| `--src_root` | 包含 HDF5 episode 文件夹的源目录 |
| `--tgt_path` | 转换后数据集的输出目录 |
| `--task_name` | 任务名称（自然语言描述） |
| `--fps` | 帧率 |
| `--robot_type` | 机器人类型标识（如 `tienkung`） |

### 模型训练

数据集转换完成后，使用 LeRobot 内置 CLI 进行训练：

```bash
export HF_LEROBOT_HOME=/path/to/datasets
lerobot-train --config_path=src/xhum/convert/configs/act_tienkung.json
```

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

修改部署脚本中的模型路径后运行：

```bash
# BrainCo 灵巧手
python -m xhum.deployment.ros2_brainco

# Inspire 灵巧手
python -m xhum.deployment.ros2_inspire
```

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
