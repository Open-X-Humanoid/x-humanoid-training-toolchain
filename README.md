
# x-humanoid training toolchain

[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Project Page](https://img.shields.io/badge/Project%20Page-RoboMIND-blue.svg)](https://x-humanoid-robomind.github.io/)
[![arXiv](https://badgen.net/badge/icon/arXiv?icon=awesome&label&color=red&style=flat-square)](https://arxiv.org/abs/2412.13877)
[![Dataset](https://img.shields.io/badge/Dataset-flopsera-000000.svg)](http://open.flopsera.com/flopsera-open/data-details/RoboMIND)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-RoboMIND-000000.svg)](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND)

**[简体中文](./README_zh.md)｜English**


This project provides a training toolchain for adapting TienKung humanoid robots and RoboMIND dataset with the open-source LeRobot framework. It enables users to  facilitates development using RoboMIND dataset and train embodied manipulation models for TienKung robots based on the Lerobot. This project lowers the barrier to entry for developing embodied manipulations while expanding the ecosystem of RoboMIND and TienKung robots.

- Support for RoboMIND, an open source multi-ontology dataset
- Upgraded to LeRobotDataset V3.0 with improved data structure and performance
- Training pipelines for TienKung robots' embodied manipulation
- Integrated MetaWorld and Libero benchmark environments
- Future roadmap for ecosystem development of RoboMIND/Huisikaiwu/TienKung


<table><tbody>

<table class="table table-striped table-bordered table-vcenter"/>
    <tbody>
    <tr><th> Title </th> <th>Description</th>
    <tr>
       <td align="center" > <a href="https://github.com/x-humanoid-robomind/x-humanoid-robomind.github.io">RoboMIND</a></td>
        <td>  a comprehensive dataset featuring 107k real-world demonstration trajectories spanning 479 distinct tasks and involving 96 unique object classes.<br></a></td>
     <tr>
         <td align="center" > <a href="https://github.com/x-humanoid-robomind/TienKung_URDF">TienKung_URDF</a></td>
        <td>urdf publish is the URDF package for Tien Kung,which includes complete robot description files (URDF) and mesh files(STL), defining core parameters such as mechanical structure, joint limits,and mass distribution.lt supports motion planning and control algorithmverification in the ROS environment and Gazebo simulation platform.<br></a></td>
    </tr>
     <tr>
          <td align="center" > <a href="https://github.com/x-humanoid-robomind/TienKung_ROS">TienKung_ROS</a></td>
        <td>The Tien Kung software system, developed based on the ROS frameworkis the low-level implementation directly responsible for hardware controlincluding key modules such as body control (body_control), robot description (robot_description), andremote control communication (usb sbus), responsible for the basic motioncontrol and hardware driving of the robot.<br></a></td>
    </tr>
    <tr>
          <td align="center" > <a href="https://github.com/x-humanoid-robomind/TienKung_Docs">TienKung_Docs</a></td>
        <td> User manuals and SDK documentation for the TienKung, including both the Lite and Pro versions, covering robot unboxing, daily usage, maintenance guidelines, and SDK interface instructions.<br></a></td>
    </tr>
    </tr>
    </tbody>
</table>

## Usage Instructions

This step converts HDF5-formatted data into the LeRobotDataset format by parsing structured observations (RGB images, joints, etc.).

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
## Training
After converting the dataset to LeRobotDataset format, users can train models using the following workflow:
- Configuration setup
Create a train_config.json file to specify the dataset path, training algorithm (e.g., ACT or Diffusion Policy), hyperparameters (learning rate, batch size), and other relevant parameters.
- Training process
With the configuration file prepared, initiate training by running the command:

```
export HF_LEROBOT_HOME=PATH_TO_LEROBOT_HOME
python lerobot/scripts/train.py --config_path=PATH_TO_CONFIG

```

## Visualization

Dataset visualization is performed using LeRobot's built-in visualization scripts.

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

## Roadmap
- Integrate more state-of-the-art robotic algorithms.
- Support for TienKung series embodied manipulation.

## Acknowledgments
RoboMIND and TienKung have adapted to the [Lerobot](https://github.com/huggingface/lerobot) framework. Thanks! 

##  Discussions 
If you're interested in RoboMIND, welcome to join our WeChat group for discussions.

<img src="./static/qrcode.png" border=0 width=30%>

