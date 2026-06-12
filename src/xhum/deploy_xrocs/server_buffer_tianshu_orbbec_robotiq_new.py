"""Server 端：获取机器人观测 → 发送给 inference client → 接收并执行预测动作。

73号机硬件（configuration.toml）：
- 19维 state/action = 左臂7 + 左手1 + 右臂7 + 右手1 + 头部3
- 相机：Orbbec Gemini336（head/left/right）
- 夹爪：Robotiq2f85Zmq
- 头部通过 ROS2 HeadController 独立控制
"""

import os
import socket
import struct
import time
import select
import pickle

import cv2
import h5py
import numpy as np
from pynput import keyboard

from bodyctrl_msgs.msg import CmdSetMotorPosition, SetMotorPosition

from xrocs.common.data_type import Joints
from xrocs.core.config_loader import ConfigLoader
from xrocs.core.station_loader import StationLoader
from xrocs.utils.logger.logger_loader import logger


# ========================= 常量 =========================

robot_type = "tienkung_pro"  # TODO !!

# 头部初始状态（19维state中最后3维，初始值待填入）
DEFAULT_HEAD_JOINT3 = 1.5
INITIAL_HEAD_STATE = [0.0, 0.4, DEFAULT_HEAD_JOINT3]  # TODO: 请填入真实的头部初始值

CAMERA_VIEW_NAME = "head"  # 可视化窗口监控 head / left / right

reset_flag = False


def _patch_infer_config(cfg_dict):
    """推理服务用：头部由脚本内 HeadController 控制，关闭 xrocs 内置 head 订阅避免 connect 阻塞。"""
    ctrl = cfg_dict.setdefault("robot", {}).setdefault("controller", {})
    if ctrl.get("enable_head_controller") or ctrl.get("sub_head_status"):
        ctrl["enable_head_controller"] = False
        ctrl["sub_head_status"] = False
        print("[Config] robot.controller: 关闭 xrocs 内置 head 控制，改用 HeadController")
    return cfg_dict


def _acquire_server_port(host, port):
    """启动前先占住端口，避免初始化完机器人后才发现端口被占用。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as e:
        if e.errno == 98:
            raise SystemExit(
                f"端口 {port} 已被占用。请先结束旧进程：pkill -f server_buffer_72demo.py"
            ) from e
        raise
    sock.listen(1)
    return sock


# ========================= 网络工具 =========================

def send_msg(sock, data: bytes):
    # 先发长度（4字节，网络字节序）
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)


def recv_msg(sock):
    # 先收长度
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    # 再收数据
    return recvall(sock, msg_len)

def recvall(sock, n):
    """保证收满 n 个字节"""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


# ========================= 键盘 =========================

def on_press(key):
    global reset_flag
    if key == keyboard.Key.space:   # 按 Space 触发
        reset_flag = True


# ========================= 头部控制 =========================

class HeadController:
    """通过 ROS2 发布头部 3 个关节指令（与 replay_buffer_72demo.py 一致）。"""

    def __init__(self, ros_node):
        self._publisher = ros_node.create_publisher(
            CmdSetMotorPosition, "/head/cmd_pos", 10
        )
        print("✅ 头部控制 publisher 已绑定到 xrocs 共享节点")

    def move_head(self, pos3):
        msg = CmdSetMotorPosition()

        cmd1 = SetMotorPosition()
        cmd1.name = 1
        cmd1.pos = 0.0
        cmd1.spd = 0.8
        cmd1.cur = 2.0

        cmd2 = SetMotorPosition()
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


# ========================= 机器人控制类 =========================

class Tienkung_dual_xrocs:
    def __init__(self, config_path):
        cfg_loader = ConfigLoader(config_path)
        self.cfg_dict = _patch_infer_config(cfg_loader.get_config())
        station_loader = StationLoader(self.cfg_dict)
        self.robot_station = station_loader.generate_station_handle()
        # generate_station_handle 已完成连接预热，无需再次 connect()

        grippers = self.robot_station.get_gripper_handle()
        if grippers:
            logger.success(f"夹爪已就绪: {list(grippers.keys())}")
        else:
            logger.error(
                "夹爪未初始化。请检查 configuration.toml 中 hand.type "
                "(应为 Robotiq2f85Zmq)"
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
            "/home/ubuntu/Dev/dylan_wu/data/init_pose",  # TODO !
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
                [-0.7903, 0.5278, -0.6346, -0.6005, -0.3605, -0.4887, -0.3676,
                 0.8929, -0.2742, 0.7000, 0.2687, 0.5630, 0.4885, 0.1073],
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
            "/home/ubuntu/Dev/wd/data/h5_for_init_data/station_sta1PlusH_dualArm-gripper-3cameras_73",  # TODO !
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
        print("[Reset] moving arms to init pose...", flush=True)
        _robot.reach_target_joint(home)
        print("[Reset] opening grippers...", flush=True)
        for gripper in self.robot_station.get_gripper_handle().values():
            gripper.open()
        logger.success("Resetting to tienkung_pro2 success!")


# ========================= 辅助函数 =========================

def _show_camera(obs):
    """显示相机画面（与 replay_buffer_72demo.py 一致）。"""
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

    cv2.imshow("rgb_vis", cv2.imdecode(images[camera_name], cv2.IMREAD_COLOR))
    cv2.waitKey(1)


def apply_gripper_postprocess(left_hand_action, right_hand_action):
    """夹爪后处理（与 replay_buffer_72demo.py 一致）。"""
    if right_hand_action > 0.8:
        right_hand_action = 1.0
    if left_hand_action > 0.8:
        left_hand_action = 1.0
    # right_hand_action = right_hand_action / 100.0
    # left_hand_action = left_hand_action / 100.0
    return left_hand_action, right_hand_action


def action_to_decompose(robo_xrocs, action_pred):
    """将 19 维 action 转为 robot_station.decompose_action 所需的 16 维格式。

    19 维: 左臂7 + 左手1 + 右臂7 + 右手1 + 头部3
    decompose_action 接收 16 维（不含头部），头部由 HeadController 独立控制。
    """
    left_arm_action = action_pred[:7]
    left_hand_action = action_pred[7]
    right_arm_action = action_pred[8:15]
    right_hand_action = action_pred[15]
    head_action = action_pred[16:19]

    left_hand_action, right_hand_action = apply_gripper_postprocess(
        left_hand_action, right_hand_action
    )

    # decompose_action 只接收 16 维（臂+夹爪），头部单独控制
    action_output = np.concatenate(
        [left_arm_action, [left_hand_action], right_arm_action, [right_hand_action]]
    )

    return robo_xrocs.robot_station.decompose_action(action_output), head_action


# ========================= reset =========================

def reset(robo_xrocs, repo_id, head_controller):
    """复位机器人到初始位置"""
    print("===== Resetting Robot =====")
    if robot_type == "tienkung_max":
        robo_xrocs.prepare_tienkung_max3(repo_id=repo_id)
    elif robot_type == "tienkung_pro":
        robo_xrocs.prepare_tienkung_pro2(repo_id=repo_id)
        head_controller.move_head(DEFAULT_HEAD_JOINT3)
    time.sleep(3)


# ========================= 主程序 =========================

listener = keyboard.Listener(on_press=on_press)
listener.start()

HOST = "0.0.0.0"
PORT = 9000
sock = _acquire_server_port(HOST, PORT)

# 初始化 ROS + XROCS
config_path = '/home/ubuntu/Documents/configuration.toml'
tienkung_dual_xrocs = Tienkung_dual_xrocs(config_path)
ros_node = tienkung_dual_xrocs.robot_station.node
head_controller = HeadController(ros_node)

print("Reset robot to initial pose ...", flush=True)

# =================================================================
# 修改这里的 repo_id 和 prompt
# TODO
repo_id = "tianshu_dualArm_73_grab_and_flip_label_up_white_bag_barcode_upward_20260610"  # 修改1 # for init state
# TODO
prompt = "The left arm picks the parcel from the recess, places it on the table and checks the tracking number. Flip it if the number faces down, then the right arm pushes the parcel with the number facing up onto the conveyor belt."  # 修改2
# ==================================================================

if robot_type == "tienkung_max":
    tienkung_dual_xrocs.prepare_tienkung_max3(repo_id=repo_id)
elif robot_type == "tienkung_pro":
    tienkung_dual_xrocs.prepare_tienkung_pro2(repo_id=repo_id)
    head_controller.move_head(DEFAULT_HEAD_JOINT3)

print(f"xrocs-server listening on port {PORT}", flush=True)
conn, addr = sock.accept()
print("Connected by", addr, flush=True)

# 用于缓存从 client 收到但尚未执行的动作
pending_actions = []

# 上一次已执行的动作，用于平滑当前动作
last_action = None

# 动作平滑系数（越接近 1 越"粘滞"/保留之前动作，0 表示不平滑）
SMOOTH_ALPHA = 0.7  # TODO !!
head_state = np.array(INITIAL_HEAD_STATE, dtype=np.float32)

while True:
    # 1. 获取机器人观测
    obs = tienkung_dual_xrocs.robot_station.get_obs()

    _show_camera(obs)

    # 按 space 复位
    if reset_flag:
        # 先发送reset信号给client
        reset_signal = {"type": "reset"}
        send_msg(conn, pickle.dumps(reset_signal))
        # 等待client确认
        while True:
            ack_data = recv_msg(conn)
            if not ack_data:
                print("[Server] No data while waiting reset_ack, client may be disconnected.")
                break
            ack = pickle.loads(ack_data)
            # 可能先收到的是之前缓冲的动作列表，直接丢弃
            if isinstance(ack, list):
                print(f"[Server] Discard buffered actions ({len(ack)}) while waiting reset_ack")
                continue
            # 真正的 reset_ack
            if isinstance(ack, dict) and ack.get("type") == "reset_ack":
                print("[Server] Client reset acknowledged")
                break

        # 清空本地动作缓存与上一次动作，避免 reset 后还执行旧命令
        pending_actions.clear()
        last_action = None

        reset(tienkung_dual_xrocs, repo_id=repo_id, head_controller=head_controller)
        reset_flag = False
        continue

    # 压缩所有相机图像
    compressed_images = {}
    for cam_name, img in obs["images"].items():
        if len(img.shape) == 1:
            compressed_images[cam_name] = img.tobytes()
        else:
            success, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            compressed_images[cam_name] = buf.tobytes()

    # 组装19维state: 左臂(7) + 左手(1) + 右臂(7) + 右手(1) + 头部(3)
    # 头部值：初始时为 head_state，收到模型预测后更新为模型输出
    state_19 = np.concatenate([
        obs['arm_joints']['left'],     # 7
        obs['hand_joints']['left'],    # 1
        obs['arm_joints']['right'],    # 7
        obs['hand_joints']['right'],   # 1
        head_state,                     # 3
    ]).astype(np.float32)
    # print("state_19", state_19)

    obs_data = {
        "state": state_19,
        "images": compressed_images,
        "prompt": prompt,
    }

    data = pickle.dumps(obs_data)
    send_msg(conn, data)

    # 2. 非阻塞方式尝试接收预测动作（可能是单个action或action chunk，也可能暂时没有）
    readable, _, _ = select.select([conn], [], [], 0.0)
    if readable:
        data = recv_msg(conn)
        if data is None:
            print("Client disconnected.")
            break

        actions_list = pickle.loads(data)
        actions = np.array(actions_list, dtype=np.float32)

        print(f"===== Received {len(actions)} action(s) from client =====")

        # 将新到达的动作追加到待执行队列
        pending_actions.extend(list(actions))

    # 3. 每个循环最多执行一个动作，保持大约 30Hz
    if len(pending_actions) > 0:
        action_pred = pending_actions.pop(0)

        # 对动作做一次与上一动作的指数平滑，使多个 buffer 之间衔接更流畅
        if last_action is not None:
            smoothed_action = last_action * SMOOTH_ALPHA + action_pred * (1.0 - SMOOTH_ALPHA)
        else:
            smoothed_action = action_pred
        last_action = smoothed_action

        # print("action_19", smoothed_action)
        # 19维action: 左臂(7) + 左手(1) + 右臂(7) + 右手(1) + 头部(3)
        action_dict, head_action = action_to_decompose(tienkung_dual_xrocs, smoothed_action)

        # 更新头部状态为模型预测值（用于下一帧观测发送）
        head_state[:] = head_action
        # print("head_state", head_state)

        # 独立控制头部
        head_controller.move_head(float(head_action[2]))

        print(f"[Server] Executing action... pending_actions_left={len(pending_actions)}")
        # print(f"  head_action: {head_action}")
        obs = tienkung_dual_xrocs.robot_station.step(action_dict)

    # 控制频率约 30Hz
    time.sleep(0.0153)

conn.close()
