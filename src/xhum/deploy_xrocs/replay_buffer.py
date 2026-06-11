"""从 HDF5 轨迹回放机器人动作，控制逻辑参考 server_buffer.py。"""

import argparse
import os
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
from pynput import keyboard

from bodyctrl_msgs.msg import CmdSetMotorPosition, SetMotorPosition
from sensor_msgs.msg import CompressedImage, Image

from xrocs.common.data_type import Joints
from xrocs.core.config_loader import ConfigLoader
from xrocs.core.station_loader import StationLoader
from xrocs.utils.logger.logger_loader import logger

robot_type = "tienkung_pro"

reset_flag = False
step_flag = False
SMOOTH_ALPHA = 0.7  # TODO !!
DEFAULT_HEAD_JOINT3 = 1.5
CAMERA_VIEW_NAME = "head" # head / left / right
# RealSense D405 腕部相机 ROS namespace，与 configuration.toml 中 camera_name 一致
_WRIST_CAMERA_NAMES = {
    "left": "camera1/camera1",
    "right": "camera2/camera2",
}

# configuration.toml 中的类型名与 xrocs 已注册类型不一致时的映射
_HAND_TYPE_ALIASES = {
    "InspireGripperEg2Ros2": "InspireGripperRos2",
}
# TienKung2Ros2Station 腕部相机仅支持 OrbbecCameraRos2（按 camera_name 订阅 ROS topic）。
# 实际硬件为 RealSense D405 时，保持 camera_name=camera1/camera1 等 topic 即可。
_WRIST_CAMERA_TYPE_ALIASES = {
    "RealSenseCameraRos2": "OrbbecCameraRos2",
    "RealsenseCameraRos2": "OrbbecCameraRos2",
    "RealsenseCameraROS2": "OrbbecCameraRos2",
}


def _patch_replay_config(cfg_dict):
    """修正 replay 所需的 xrocs 配置类型名，不修改 xrocs 包本身。"""
    for side in ("left", "right"):
        hand_cfg = cfg_dict.get("hand", {}).get(side, {})
        old_type = hand_cfg.get("type")
        if old_type in _HAND_TYPE_ALIASES:
            new_type = _HAND_TYPE_ALIASES[old_type]
            hand_cfg["type"] = new_type
            print(f"[Config] hand.{side}.type: {old_type} -> {new_type}")

    for side in ("left", "right"):
        camera_cfg = cfg_dict.get("camera", {}).get(side, {})
        if not camera_cfg.get("enable", False):
            continue
        old_type = camera_cfg.get("type")
        if old_type in _WRIST_CAMERA_TYPE_ALIASES:
            new_type = _WRIST_CAMERA_TYPE_ALIASES[old_type]
            camera_cfg["type"] = new_type
            print(
                f"[Config] camera.{side}.type: {old_type} -> {new_type} "
                f"(topic={camera_cfg.get('camera_name')})"
            )

    return cfg_dict


class HeadController:
    """复用 xrocs 共享 ROS 2 节点发布头部指令，避免重复 rclpy.init。"""

    def __init__(self, ros_node):
        self._publisher = ros_node.create_publisher(
            CmdSetMotorPosition, "/head/cmd_pos", 10
        )
        print("✅ 头部控制 publisher 已绑定到 xrocs 共享节点")

    def move_head(self, pos3):
        msg = CmdSetMotorPosition()
        msg.cmds = []

        cmd1 = SetMotorPosition() # 本关节不动
        cmd1.name = 1
        cmd1.pos = 0.0
        cmd1.spd = 0.8
        cmd1.cur = 2.0

        cmd2 = SetMotorPosition() # 本关节不动
        cmd2.name = 2
        cmd2.pos = 0.40
        cmd2.spd = 0.8
        cmd2.cur = 2.0

        cmd3 = SetMotorPosition()
        cmd3.name = 3
        cmd3.pos = pos3
        cmd3.spd = 0.8
        cmd3.cur = 2.0

        msg.cmds = [cmd1, cmd2, cmd3]
        self._publisher.publish(msg)


class WristCameraTopicRelay:
    """RealSense D405 发布 image_rect_raw，xrocs 订阅 image_raw，启动时做 topic 转发。"""

    def __init__(self, ros_node, cfg_dict):
        self._subs = []
        for side, cam_name in _WRIST_CAMERA_NAMES.items():
            camera_cfg = cfg_dict.get("camera", {}).get(side, {})
            if not camera_cfg.get("enable", False):
                continue
            self._relay_rgb_compressed(ros_node, cam_name)
            if not camera_cfg.get("use_compress_depth", False):
                self._relay_depth_raw(ros_node, cam_name)

    def _relay_rgb_compressed(self, ros_node, cam_name):
        src = f"/{cam_name}/color/image_rect_raw/compressed"
        dst = f"/{cam_name}/color/image_raw/compressed"
        pub = ros_node.create_publisher(CompressedImage, dst, 10)
        sub = ros_node.create_subscription(CompressedImage, src, pub.publish, 10)
        self._subs.append(sub)
        print(f"[CameraRelay] {src} -> {dst}")

    def _relay_depth_raw(self, ros_node, cam_name):
        src = f"/{cam_name}/depth/image_rect_raw"
        dst = f"/{cam_name}/depth/image_raw"
        pub = ros_node.create_publisher(Image, dst, 10)
        sub = ros_node.create_subscription(Image, src, pub.publish, 10)
        self._subs.append(sub)
        print(f"[CameraRelay] {src} -> {dst}")


def on_press(key):
    global reset_flag, step_flag
    if key == keyboard.Key.space:
        reset_flag = True
    elif key == keyboard.Key.enter:
        step_flag = True


def reset(robo_xrocs, repo_id, head_controller):
    print("===== Resetting Robot =====")
    if robot_type == "tienkung_max":
        robo_xrocs.prepare_tienkung_max3(repo_id=repo_id)
    elif robot_type == "tienkung_pro":
        robo_xrocs.prepare_tienkung_pro2(repo_id=repo_id)
        head_controller.move_head(DEFAULT_HEAD_JOINT3)
    time.sleep(3)


class Tienkung_dual_xrocs:
    def __init__(self, config_path):
        cfg_loader = ConfigLoader(config_path)
        self.cfg_dict = _patch_replay_config(cfg_loader.get_config())
        station_loader = StationLoader(self.cfg_dict)
        self.robot_station = station_loader.generate_station_handle()
        self.robot_station.connect()

        grippers = self.robot_station.get_gripper_handle()
        if grippers:
            logger.success(f"夹爪已就绪: {list(grippers.keys())}")
        else:
            logger.error(
                "夹爪未初始化。请检查 configuration.toml 中 hand.type "
                "(应为 InspireGripperRos2，而非 InspireGripperEg2Ros2)"
            )

        cameras = self.robot_station.get_camera_handle()
        expected_cameras = [
            side
            for side in ("head", "left", "right")
            if self.cfg_dict.get("camera", {}).get(side, {}).get("enable", False)
        ]
        missing_cameras = [side for side in expected_cameras if side not in cameras]
        if missing_cameras:
            logger.error(f"相机未初始化: {missing_cameras}")
        else:
            logger.success(f"相机已就绪: {list(cameras.keys())}")

    def prepare_tienkung_max3(self, task_h5_init_pose_dir=None, repo_id=None):
        print("preparing tienkung_max3")
        _robot = self.robot_station.get_robot_handle()["robot"]

        task_h5_init_pose_dir = os.path.join(
            "/home/ubuntu/Dev/dylan_wu/data/init_pose",
            repo_id,
        )

        if task_h5_init_pose_dir is not None:
            task_h5_init_pose_path = task_h5_init_pose_dir + "/trajectory.hdf5"
            print(f"task_h5_init_pose_path: {task_h5_init_pose_path}")

            with h5py.File(task_h5_init_pose_path, "r") as h5_file:
                init_pose = h5_file["puppet"]["arm_joint_position"][0]
            print("init_pose shape", init_pose.shape)
            home = Joints(init_pose, num_of_dofs=14)
        else:
            home = Joints(
                [
                    -0.7903, 0.5278, -0.6346, -0.6005, -0.3605, -0.4887, -0.3676,
                    0.8929, -0.2742, 0.7000, 0.2687, 0.5630, 0.4885, 0.1073,
                ],
                num_of_dofs=14,
            )
        _robot.reach_target_joint(home)
        print("Robot Home:", home)
        for gripper in self.robot_station.get_gripper_handle().values():
            gripper.open()
        logger.success("Resetting to tienkung_max3 success!")

    def prepare_tienkung_pro2(self, task_h5_init_pose_dir=None, repo_id=None):
        print("preparing tienkung_pro2")
        _robot = self.robot_station.get_robot_handle()["robot"]

        task_h5_init_pose_dir = os.path.join(
            "/home/ubuntu/Dev/dylan_wu/data/h5_for_init_data/h5_data_for_init/station_sta1PlusH_dualArm-gripper-3cameras_72", # TODO !
            repo_id,
        )

        if task_h5_init_pose_dir is not None:
            task_h5_init_pose_path = task_h5_init_pose_dir + "/trajectory.hdf5"
            print(f"task_h5_init_pose_path: {task_h5_init_pose_path}")

            with h5py.File(task_h5_init_pose_path, "r") as h5_file:
                init_pose_left = h5_file["puppet"]["arm_left_position_align"]["data"][0]
                init_pose_right = h5_file["puppet"]["arm_right_position_align"]["data"][0]
                init_pose = np.concatenate([init_pose_left, init_pose_right])
            print("init_pose shape", init_pose.shape)
            home = Joints(init_pose, num_of_dofs=14)
        else:
            home = Joints([0, 0, 0, -1.7, 0, 0, 0, 0, 0, 0, -1.7, 0, 0, 0], num_of_dofs=14)

        time.sleep(2)
        _robot.reach_target_joint(home)
        for gripper in self.robot_station.get_gripper_handle().values():
            gripper.open()
        logger.success("Resetting to tienkung_pro2 success!")


def _extract_dim(arr, axis=-1, index=0):
    """沿指定 axis 提取 index 位置的数据，并转为 float32。"""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        return arr
    if axis < 0:
        axis += arr.ndim
    if axis >= arr.ndim:
        return arr
    if arr.shape[axis] == 1 and index == 0:
        return np.squeeze(arr, axis=axis)
    return np.take(arr, index, axis=axis)


def load_actions_from_h5(h5_path: Path, start: int = 0, end: int | None = None):
    """从 HDF5 读取 puppet 轨迹，组装为 17 维动作序列（16 维臂/夹爪 + 1 维头部）。"""
    required_keys = [
        "puppet/arm_left_position_align/data",
        "puppet/arm_right_position_align/data",
        "puppet/end_effector_left_position_align/data",
        "puppet/end_effector_right_position_align/data",
    ]

    with h5py.File(h5_path, "r") as f:
        for key in required_keys:
            if key not in f:
                raise KeyError(f"H5 缺少必要键: {key} (file={h5_path})")

        # arm_left = np.array(f["puppet/arm_left_position_align/data"])
        # arm_right = np.array(f["puppet/arm_right_position_align/data"])
        # gripper_left = _extract_dim(f["puppet/end_effector_left_position_align/data"], axis=-1, index=0)
        # gripper_right = _extract_dim(f["puppet/end_effector_right_position_align/data"], axis=-1, index=0)

        arm_left = np.array(f["puppet/arm_left_position_align/data"]) # master
        arm_right = np.array(f["puppet/arm_right_position_align/data"]) # master
        gripper_left = _extract_dim(f["puppet/end_effector_left_position_align/data"], axis=-1, index=0)
        gripper_right = _extract_dim(f["puppet/end_effector_right_position_align/data"], axis=-1, index=0)

        num_frames = len(arm_left)
        head_key = "puppet/head_position_align/data"
        if head_key in f:
            head_pose = _extract_dim(f[head_key], axis=1, index=2)
        else:
            print(f"⚠️ H5 缺少 {head_key}，头部关节使用默认值 {DEFAULT_HEAD_JOINT3}")
            head_pose = np.full(num_frames, DEFAULT_HEAD_JOINT3, dtype=np.float32)

        if end is None or end < 0:
            end = num_frames
        end = min(end, num_frames)
        start = max(0, start)
        if start >= end:
            raise ValueError(f"无效帧范围: start={start}, end={end}, total={num_frames}")

        arm_left = arm_left[start:end]
        arm_right = arm_right[start:end]
        gripper_left = gripper_left[start:end]
        gripper_right = gripper_right[start:end]
        head_pose = head_pose[start:end]

        actions = np.concatenate(
            [
                arm_left,
                gripper_left.reshape(-1, 1),
                arm_right,
                gripper_right.reshape(-1, 1),
                head_pose.reshape(-1, 1),
            ],
            axis=1,
        )

        timestamps = None
        if "timestamp" in f:
            t = np.array(f["timestamp"])
            if len(t) == num_frames:
                timestamps = t[start:end]

    print(f"✅ 已加载 H5 轨迹: {h5_path}")
    print(f"   帧范围 [{start}, {end}), 共 {len(actions)} 帧, action_dim={actions.shape[1]}")
    return actions, timestamps


# TODO
def apply_gripper_postprocess(left_hand_action, right_hand_action):
    """与 server_buffer.py 保持一致的夹爪后处理。"""
    # if right_hand_action > 0.99:
    #     right_hand_action = 1.0
    # if left_hand_action > 0.2:
    #     left_hand_action += 0.1
    right_hand_action = right_hand_action / 100.0
    left_hand_action = left_hand_action / 100.0
    return left_hand_action, right_hand_action


def _show_camera(obs):
    if robot_type == "tienkung_max":
        camera_name = "ob_camera_head"
    elif robot_type == "tienkung_pro":
        camera_name = CAMERA_VIEW_NAME
    else:
        return

    images = obs.get("images", {})
    if camera_name not in images:
        print(f"⚠️ 相机 '{camera_name}' 无图像，可用: {list(images.keys())}")
        return

    cv2.imshow(
        "rgb_vis",
        cv2.imdecode(images[camera_name], cv2.IMREAD_COLOR),
    )
    cv2.waitKey(1)


def wait_for_enter(robo_xrocs, frame_idx, total_frames):
    """等待回车键；等待期间仅刷新相机画面，不发送运动指令。"""
    global step_flag, reset_flag

    print(f"[Replay] 按 Enter 执行第 {frame_idx + 1}/{total_frames} 帧 (Space 复位)")
    step_flag = False

    while not step_flag:
        if reset_flag:
            return False

        obs = robo_xrocs.robot_station.get_obs()
        _show_camera(obs)
        time.sleep(0.03)

    step_flag = False
    return True


def action_to_robot_dict(robo_xrocs, action_pred):
    left_arm_action = action_pred[:7]
    left_hand_action = action_pred[7]
    right_arm_action = action_pred[8:15]
    right_hand_action = action_pred[15]


    left_hand_action, right_hand_action = apply_gripper_postprocess(
        left_hand_action, right_hand_action
    )

    action_output = np.concatenate(
        [left_arm_action, [left_hand_action], right_arm_action, [right_hand_action]]
    )

    print("left hand: ", left_hand_action)
    print("right hand: ", right_hand_action)
    
    return robo_xrocs.robot_station.decompose_action(action_output)


def replay_trajectory(
    robo_xrocs,
    actions,
    smooth_alpha=SMOOTH_ALPHA,
    loop=False,
    head_controller=None,
):
    """逐步回放动作序列，每帧需按 Enter 才发送运动指令。"""
    if len(actions) == 0:
        print("⚠️ 动作序列为空，跳过回放")
        return "done"

    last_action = None
    frame_idx = 0

    while frame_idx < len(actions):
        if reset_flag:
            return "reset"

        if not wait_for_enter(robo_xrocs, frame_idx, len(actions)):
            return "reset"

        action_pred = actions[frame_idx]

        # if last_action is not None and smooth_alpha > 0:
        #     smoothed_action = last_action * smooth_alpha + action_pred * (1.0 - smooth_alpha)
        # else:
        #     smoothed_action = action_pred
        # last_action = smoothed_action

        # action_dict = action_to_robot_dict(robo_xrocs, smoothed_action)

        action_dict = action_to_robot_dict(robo_xrocs, action_pred)
        robo_xrocs.robot_station.step(action_dict)
        print("action_pred: ", action_pred)
        if head_controller is not None:
            head_controller.move_head(float(action_pred[16]))
        print("head_pose: ", float(action_pred[16]))
        print(f"[Replay] 已执行 frame {frame_idx + 1}/{len(actions)}")

        obs = robo_xrocs.robot_station.get_obs()
        _show_camera(obs)

        frame_idx += 1

    if loop:
        return "loop"
    return "done"


def parse_args():
    parser = argparse.ArgumentParser(description="从 HDF5 回放 tienkung 机器人轨迹")
    parser.add_argument(
        "--h5",
        "-f",
        type=str,
        required=True,
        help="HDF5 轨迹文件路径 (trajectory.hdf5)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/ubuntu/Documents/configuration.toml",
        help="XROCS 配置文件路径",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="tienkung_station_16_take_bread_out_of_microwave_to_desktop_no_hand_251129",
        help="reset 时使用的初始位姿 repo_id",
    )
    parser.add_argument("--start", type=int, default=0, help="起始帧 (含)")
    parser.add_argument("--end", type=int, default=-1, help="结束帧 (不含), -1 表示到末尾")
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=SMOOTH_ALPHA,
        help="动作平滑系数，与 server_buffer.py 一致",
    )
    parser.add_argument("--loop", action="store_true", help="回放结束后自动循环")
    parser.add_argument(
        "--no-init-reset",
        action="store_true",
        help="启动时不执行 prepare_* 初始复位",
    )
    parser.add_argument(
        "--camera-view",
        type=str,
        choices=["head", "left", "right"],
        default="head",
        help="回放时显示的相机视角 (head/left/right)",
    )
    return parser.parse_args()


def main():
    global CAMERA_VIEW_NAME
    args = parse_args()
    CAMERA_VIEW_NAME = args.camera_view
    h5_path = Path(args.h5)
    if not h5_path.is_file():
        raise FileNotFoundError(f"未找到 H5 文件: {h5_path}")

    actions, _ = load_actions_from_h5(h5_path, start=args.start, end=args.end)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    tienkung_dual_xrocs = Tienkung_dual_xrocs(args.config)
    ros_node = tienkung_dual_xrocs.robot_station.node
    WristCameraTopicRelay(ros_node, tienkung_dual_xrocs.cfg_dict)
    head_controller = HeadController(ros_node)

    try:
        if not args.no_init_reset:
            print("Reset robot to initial pose ...")
            reset(tienkung_dual_xrocs, args.repo_id, head_controller)

        print("===== H5 Replay Started =====")
        print("按 Enter 逐步执行下一帧，按 Space 复位并重新回放")

        global reset_flag, step_flag
        while True:
            step_flag = False
            status = replay_trajectory(
                tienkung_dual_xrocs,
                actions,
                smooth_alpha=args.smooth_alpha,
                loop=args.loop,
                head_controller=head_controller,
            )

            if status == "reset":
                reset_flag = False
                step_flag = False
                reset(tienkung_dual_xrocs, args.repo_id, head_controller)
                print("[Replay] 复位完成，重新开始回放...")
                continue

            if status == "loop":
                print("[Replay] 循环回放...")
                continue

            print("[Replay] 回放完成")
            break
    finally:
        cv2.destroyAllWindows()
        listener.stop()
        print("===== Replay Closed =====")


if __name__ == "__main__":
    main()
