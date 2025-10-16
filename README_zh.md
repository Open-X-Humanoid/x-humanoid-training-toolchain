# x-humanoid training toolchain

[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Project Page](https://img.shields.io/badge/Project%20Page-RoboMIND-blue.svg)](https://x-humanoid-robomind.github.io/)
[![arXiv](https://badgen.net/badge/icon/arXiv?icon=awesome&label&color=red&style=flat-square)](https://arxiv.org/abs/2412.13877)
[![Dataset](https://img.shields.io/badge/Dataset-flopsera-000000.svg)](http://open.flopsera.com/flopsera-open/data-details/RoboMIND)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-RoboMIND-000000.svg)](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND)

**[English](./README.md)｜简体中文**

## 项目介绍
本项目是 RoboMIND 数据集和天工机器人对于 Lerobot 开源框架适配的训练工具链。使用此项目，用户可以基于 Lerobot 开源框架的算法实现在 RoboMIND 数据集的使用和天工机器人的具身操作，降低开发者开发门槛，扩展 RoboMIND 数据集和天工机器人生态建设。

- 支持开源多本体数据集 RoboMIND。
- 升级至 LeRobotDataset V3.0，提升数据结构和性能。
- 支持天工机器人的具身操作训练。
- 集成 MetaWorld 和 Libero 基准测试环境。
- 未来支持 RoboMIND/慧思开物/天工的生态建设。

<table><tbody>

<table class="table table-striped table-bordered table-vcenter"/>
    <tbody>
    <tr><th> Title </th> <th>Description</th>
    <tr>
       <td align="center" > <a href="https://github.com/x-humanoid-robomind/x-humanoid-robomind.github.io">RoboMIND数据集</a></td>
        <td>  RoboMIND数据集汇集了多种机器人平台的操作数据,该数据集包含了在479种不同任务中涉及96类独特物体的10.7万条真实世界演示轨迹。
<br></a></td>
     <tr>
         <td align="center" > <a href="https://github.com/x-humanoid-robomind/TienKung_URDF">天工URDF</a></td>
        <td> 包含了完整的机器人描述文件 (URDF) 和网格文件 (STL)，定义了机械结构、关节限位、质量分布等核心参数。支持在 ROS 环境和 Gazebo 仿真平台中进行运动规划和控制算法验证。<br></a></td>
    </tr>
     <tr>
          <td align="center" > <a href="https://github.com/x-humanoid-robomind/TienKung_ROS">天工软件系统</a></td>
        <td>基于 ROS 框架开发，作为直接负责硬件控制的底层实现，包含了本体控制 (body_control)、机器人描述 (robot_description)、遥控器通信 (usb_sbus) 等关键模块，负责机器人的基础运动控制和硬件驱动。<br></a></td>
    </tr>
    <tr>
          <td align="center" > <a href="https://github.com/x-humanoid-robomind/TienKung_Docs">天工文档</a></td>
        <td> 天工通用人形机器人的用户手册、SDK 文档，包括 Lite 版本 和 Pro 版本，涵盖机器人开箱、日常使用、维护指南、SDK 接口说明，帮助用户和开发者更高效地使用和开发天工机器人。<br></a></td>
    </tr>
    </tr>
    </tbody>
</table>

## 使用说明
读取 hdf5 格式的数据，根据定义的 keys 解析 observations（images, joints, etc.）。

```
cd scripts
sh convert.sh #Modify the path and args. 

```

```
--config
Description: Path to the configuration JSON file containing settings for the application.
--repo_id
Description: ID for the dataset.
--src_root
Description: Source directory containing raw input data files.
--tgt_path
Description: Target directory path for processed output files.
--task_name
Description: Identifier for the current processing task.
--fps
Description: Frames per second setting for video processing operations.
--robot_type
Description: Identifier for robot hardware platform.

```

## 模型训练
将数据集转换为 LeRobot 的格式后，用户可以使用以下工作流程进行模型训练：
- 训练配置。
创建一个 train_config. json 文件来指定数据集路径、训练算法（例如，ACT或扩散策略）、超参数（学习率、批大小）和其他相关参数。
- 创建配置文件后，用户可以通过执行以下命令开始模型训练：

```
export HF_LEROBOT_HOME=PATH_TO_LEROBOT_HOME
python lerobot/scripts/train.py --config_path=PATH_TO_CONFIG

```

## 可视化

使用 LeRobot 的本地工具进行数据集的可视化。

```
python lerobot/scripts/visualize_dataset.py --repo-id ID --episode-index 0 --root PATH_TO_ROOT

```

<div style="display: flex;">
  <img src="./static/demo1.gif" width="300">
  <img src="./static/demo2.gif" width="300">
</div>
</div>

<div style="display: flex;">
  <img src="./static/demo3.gif" width="300">
  <img src="./static/demo4.gif" width="300">
</div>
</div>

## 计划
- 未来支持更多前沿算法。
- 支持天工系列的具身操作能力。

## 致谢
RoboMIND和天工机器人已经适配了[Lerobot](https://github.com/huggingface/lerobot)，非常感谢！

##  讨论
如果您对 RoboMIND 有兴趣, 欢迎加入我们的社群进行讨论

<img src="./static/qrcode.png" border=0 width=30%>