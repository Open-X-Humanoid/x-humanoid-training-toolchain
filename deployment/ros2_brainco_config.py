#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import cv2
from typing import Tuple
import time
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
import h5py
import yaml
import argparse
import threading

# 导入 BrainCo 手部特有的消息类型
from ros2_stark_interfaces.msg import MotorStatus, SetMotorMulti

# 导入机械臂控制消息类型
try:
    from bodyctrl_msgs.msg import CmdSetMotorPosition, MotorStatusMsg, SetMotorPosition
    BODYCTRL_AVAILABLE = True
except ImportError:
    from std_msgs.msg import String as CmdSetMotorPosition
    from std_msgs.msg import String as MotorStatusMsg
    from std_msgs.msg import String as SetMotorPosition
    BODYCTRL_AVAILABLE = False
    print("Warning: bodyctrl_msgs package is not available, will use String message type as replacement")

# 使用 message_filters 进行消息同步
from message_filters import ApproximateTimeSynchronizer, Subscriber
from action_policy import PolicyAgent

class PolicyAgentNode(Node):
    def __init__(self, config_path=None):
        # 初始化节点
        super().__init__('policy_agent_node_brainco')
        
        # 加载配置
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config_brainco.yaml')
        
        self.config = self.load_config(config_path)
        self.mode = self.config.get('mode', 'model')  # 'model' 或 'replay'
        
        # DEBUG 计数器
        self.cnt = 0
        
        # 根据模式初始化
        if self.mode == 'model':
            model_path = self.config.get('model_path', 'PATH_TO_MODEL')
            self.action_policy = PolicyAgent(model_path)
            self.get_logger().info(f'Initialized in MODEL mode with model: {model_path}')
        elif self.mode == 'replay':
            self.h5_path = self.config.get('h5_path', 'PATH_TO_H5')
            self.action_policy = None
            self.get_logger().info(f'Initialized in REPLAY mode with HDF5: {self.h5_path}')
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'model' or 'replay'")
        
        # 初始化 BrainCo 手部控制发布者
        self.left_hand_publisher = self.create_publisher(
            SetMotorMulti,
            '/left_hand/set_motor_multi',
            10)
            
        self.right_hand_publisher = self.create_publisher(
            SetMotorMulti,
            '/right_hand/set_motor_multi',
            10)
            
        # 订阅 BrainCo 手部状态
        self.left_hand_subscription = self.create_subscription(
            MotorStatus,
            '/left_hand/motor_status',
            self.left_hand_callback,
            10)
        
        self.right_hand_subscription = self.create_subscription(
            MotorStatus,
            '/right_hand/motor_status',
            self.right_hand_callback,
            10)
            
        # 左手和右手的当前关节位置 (BrainCo 通常为 6 维)
        self.left_hand_pos = np.zeros(6)
        self.right_hand_pos = np.zeros(6)
        
        # 订阅机械臂状态
        self.joint_state_sub = self.create_subscription(
            MotorStatusMsg,
            '/arm/status',
            self.arm_callback,
            10)
            
        self.left_jpos = None
        self.right_jpos = None
        
        # 初始化 CV Bridge
        self.bridge = CvBridge()
        self.image = None
        self.depth = None
        
        # 设置相机名称
        camera_name = self.config.get('camera_name', 'camera')
        
        # 使用 message_filters 同步图像
        self.rgb_sub = Subscriber(self, Image, f'/{camera_name}/color/image_raw')
        self.depth_sub = Subscriber(self, Image, f'/{camera_name}/depth/image_raw')
        
        # 创建同步器
        self.ats = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ats.registerCallback(self.image_callback)
        
        # 创建机械臂控制发布者
        self.dual_arm_controller = self.create_publisher(
            CmdSetMotorPosition,
            '/arm/cmd_pos',
            10)
            
        self.get_logger().info('PolicyAgentNode initialization completed')
    
    def load_config(self, config_path):
        """从 YAML 文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.get_logger().info(f'Configuration loaded from: {config_path}')
            return config
        except Exception as e:
            self.get_logger().error(f'Error loading config: {e}, using defaults')
            return self.get_default_config()
    
    def get_default_config(self):
        """返回默认配置"""
        return {
            'mode': 'model',
            'model_path': 'PATH_TO_MODEL',
            'camera_name': 'camera',
            'action_rate': 20.0,
        }
    
    def load_hdf5_actions(self, h5_path: str) -> list[np.ndarray]:
        """从 HDF5 加载动作序列"""
        try:
            with h5py.File(h5_path, "r") as f:
                left_arm = np.asarray(f["puppet/arm_left_position_align/data"])
                right_arm = np.asarray(f["puppet/arm_right_position_align/data"])
                left_hand = np.asarray(f["puppet/end_effector_left_position_align/data"])
                right_hand = np.asarray(f["puppet/end_effector_right_position_align/data"])
            T = min(left_arm.shape[0], right_arm.shape[0], left_hand.shape[0], right_hand.shape[0])
            actions = []
            for t in range(T):
                act = np.concatenate([left_arm[t], left_hand[t], right_arm[t], right_hand[t]]).astype(np.float64)
                actions.append(act)
            return actions
        except Exception as e:
            self.get_logger().error(f"Failed to load HDF5: {e}")
            return []
    
    def left_hand_callback(self, msg):
        # 处理 BrainCo 左手状态
        self.left_hand_pos = np.array(msg.positions)
    
    def right_hand_callback(self, msg):
        # 处理 BrainCo 右手状态
        self.right_hand_pos = np.array(msg.positions)
    
    def arm_callback(self, msg):
        tmp_arms_status = [val.pos for val in msg.status]
        self.left_jpos = tmp_arms_status[:7]
        self.right_jpos = tmp_arms_status[7:]
    
    def get_current_arm_status(self):
        if self.left_jpos is None or self.right_jpos is None:
            return np.zeros(14)
        return np.concatenate([self.left_jpos, self.right_jpos])
    
    def image_callback(self, rgb_msg, depth_msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            self.depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
   
    def get_current_imgs(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.image, self.depth

    def get_current_hand_position(self, hand_type='left'):
        return self.left_hand_pos if hand_type == 'left' else self.right_hand_pos

    def get_current_preprospective(self):
        arm_status = self.get_current_arm_status()
        left_hand = self.get_current_hand_position('left').flatten()
        right_hand = self.get_current_hand_position('right').flatten()
        return np.concatenate([arm_status[:7], left_hand, arm_status[7:], right_hand])

    def _construct_dual_arm_ctrl_msg(self, target_joint: list[float]):
        msg = CmdSetMotorPosition()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        for idx, val in enumerate(target_joint):
            cmd = SetMotorPosition()
            cmd.name = (11 + idx) if idx < 7 else (14 + idx - 7)
            cmd.pos = val.item()
            cmd.spd, cmd.cur = 0.5, 5.0 # 仿照原版默认值
            msg.cmds.append(cmd)
        return msg
    
    def reach_target_joint(self, target_joint, asynchronous: bool = False) -> bool:
        fine_step = 500
        current_status = self.get_current_arm_status()
        step_array = np.linspace(current_status, target_joint, fine_step)
        for stp in step_array:
            self.dual_arm_controller.publish(self._construct_dual_arm_ctrl_msg(stp))
            time.sleep(1.0/400.0)
        return True
    
    def control_hand(self, hand_type, position):
        # BrainCo 手部控制逻辑
        msg = SetMotorMulti()
        msg.mode = 1 # 位置模式
        if isinstance(position, (list, np.ndarray)):
            msg.positions = np.array(position, dtype=np.uint16).tolist()
        else:
            msg.positions = [uint16(position)] * 6
        
        if hand_type == 'left':
            self.left_hand_publisher.publish(msg)
        else:
            self.right_hand_publisher.publish(msg)

    def publish_action(self, action):
        # 仿照原版拆分动作: [arm_left(7), hand_left(6), arm_right(7), hand_right(6)]
        target_joint = np.concatenate([action[:7], action[13:20]])
        left_hand_pos = action[7:13]
        right_hand_pos = action[20:]
        
        self.dual_arm_controller.publish(self._construct_dual_arm_ctrl_msg(target_joint))
        self.control_hand('left', left_hand_pos)
        self.control_hand('right', right_hand_pos)
    
    def get_obs(self):
        obs = {'images': {'camera_head': None}, 'arm_gripper_joints': None}
        dual_arm_hand_status = self.get_current_preprospective()
        rgb, depth = self.get_current_imgs()
        if rgb is None: return None
        obs['images']['camera_head'] = rgb
        obs['arm_gripper_joints'] = dual_arm_hand_status
        return obs
    
    def reset_home(self):
        # Reset to initial position
        state_2 = [-0.05916397, 0.11694484, 0.00816471, -1.6296118, -0.18107964, -0.1322771, -0.08812793,
                  -0.00609963, 0.05809595, -0.0326848, -1.6615903, -0.15082923, 0.03735191, 0.00886455]
        
        self.get_logger().info("Resetting to initial position...")
        time.sleep(5)  # Wait for system to stabilize
        self.reach_target_joint(state_2)
        
        hand_state = [0,0,0,0,0,0]
        hand_state = [99] * 6 # Close hands
        # Control left and right hands to initial position
        self.control_hand_brainco('left', np.asarray(hand_state))
        self.control_hand_brainco('right',  np.asarray(hand_state))
        
        self.get_logger().info("Reset to initial position completed")

    def warm_up(self):
        time.sleep(3)
        self.get_logger().info('Warm-up completed')

    def run(self):
        self.warm_up()
        self.reset_home()
        
        action_rate = self.config.get('action_rate', 20.0)
        action_period = 1.0 / action_rate
        
        if self.mode == 'replay':
            actions = self.load_hdf5_actions(self.h5_path)
            for act in actions:
                self.publish_action(act)
                time.sleep(action_period)
        elif self.mode == 'model':
            while rclpy.ok():
                obs = self.get_obs()
                if obs is None:
                    time.sleep(0.1)
                    continue
                action = self.action_policy.inference(obs)
                self.publish_action(action[0].numpy())
                time.sleep(action_period)

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args_parsed, unknown = parser.parse_known_args(args)
    
    rclpy.init(args=unknown)
    node = PolicyAgentNode(config_path=args_parsed.config)
    
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    
    import threading
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()