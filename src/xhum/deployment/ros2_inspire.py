#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 deployment node for TienKung robot with Inspire dexterous hands."""

import threading
import time
from typing import Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

try:
    from bodyctrl_msgs.msg import CmdSetMotorPosition, MotorStatusMsg, SetMotorPosition

    BODYCTRL_AVAILABLE = True
except ImportError:
    from std_msgs.msg import String as CmdSetMotorPosition
    from std_msgs.msg import String as MotorStatusMsg
    from std_msgs.msg import String as SetMotorPosition

    BODYCTRL_AVAILABLE = False
    print("Warning: bodyctrl_msgs package is not available, will use String message type as replacement")

from xhum.deployment.policy_agent import PolicyAgent


class PolicyAgentNode(Node):
    def __init__(self):
        super().__init__("policy_agent_node")

        self.cnt = 0

        model_path = "PATH_TO_MODEL"
        self.action_policy = PolicyAgent(model_path)

        self.left_hand_publisher = self.create_publisher(JointState, "/inspire_hand/ctrl/left_hand", 10)
        self.right_hand_publisher = self.create_publisher(JointState, "/inspire_hand/ctrl/right_hand", 10)

        self.left_hand_subscription = self.create_subscription(
            JointState, "/inspire_hand/state/left_hand", self.left_hand_callback, 10
        )
        self.right_hand_subscription = self.create_subscription(
            JointState, "/inspire_hand/state/right_hand", self.right_hand_callback, 10
        )

        self.left_hand_pos = 0.0
        self.right_hand_pos = 0.0

        self.joint_state_sub = self.create_subscription(MotorStatusMsg, "/arm/status", self.arm_callback, 10)

        self.left_jpos = None
        self.right_jpos = None

        self.bridge = CvBridge()
        self.image = None
        self.depth = None

        camera_name = "camera"
        self.rgb_sub = Subscriber(self, Image, f"/{camera_name}/color/image_raw")
        self.depth_sub = Subscriber(self, Image, f"/{camera_name}/depth/image_raw")

        self.ats = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ats.registerCallback(self.image_callback)

        self.dual_arm_controller = self.create_publisher(CmdSetMotorPosition, "/arm/cmd_pos", 10)

        self.get_logger().info("PolicyAgentNode initialization completed")

    def left_hand_callback(self, msg):
        if len(msg.position) > 0:
            self.left_hand_pos = msg.position

    def right_hand_callback(self, msg):
        if len(msg.position) > 0:
            self.right_hand_pos = msg.position

    def arm_callback(self, msg):
        tmp_arms_status = [val.pos for val in msg.status]
        self.left_jpos = tmp_arms_status[:7]
        self.right_jpos = tmp_arms_status[7:]

    def get_current_arm_status(self):
        if self.left_jpos is None or self.right_jpos is None:
            self.get_logger().warning("Robotic arm status not ready yet")
            return np.zeros(14)
        return np.concatenate([self.left_jpos, self.right_jpos])

    def image_callback(self, rgb_msg, depth_msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            self.depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            self.get_logger().debug("Synchronized color and depth images received")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def get_current_imgs(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.image, self.depth

    def get_current_hand_position(self, hand_type="left"):
        if hand_type == "left":
            return np.array(self.left_hand_pos)
        else:
            return np.array(self.right_hand_pos)

    def get_current_preprospective(self):
        arm_status = self.get_current_arm_status()
        left_hand = self.get_current_hand_position("left").flatten()
        right_hand = self.get_current_hand_position("right").flatten()
        return np.concatenate([arm_status[:7], left_hand, arm_status[7:], right_hand])

    def _construct_dual_arm_ctrl_msg(self, target_joint: list[float]):
        msg = CmdSetMotorPosition()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()

        for idx, val in enumerate(target_joint):
            cmd = SetMotorPosition()
            cmd.name = 11 + idx if idx < 7 else 14 + idx
            cmd.pos = val.item()
            cmd.spd = 0.5
            cmd.cur = 5.0
            msg.cmds.append(cmd)
        return msg

    def reach_target_joint(self, target_joint, asynchronous: bool = False) -> bool:
        fine_step = 500
        current_status = self.get_current_arm_status()
        step_array = np.linspace(current_status, target_joint, fine_step)

        self.get_logger().info("Start moving robotic arm to target position")
        for stp in step_array:
            self.dual_arm_controller.publish(self._construct_dual_arm_ctrl_msg(stp))
            time.sleep(1.0 / 400.0)

        self.get_logger().info("Robotic arm has reached target position")
        return True

    def control_hand(self, hand_type, position):
        if isinstance(position, (list, np.ndarray)):
            position = [round(np.clip(float(pos), 0, 1), 1) for pos in position]
        else:
            position = np.clip(float(position), 0, 1)
            position = [round(position, 1)] * 6

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["1", "2", "3", "4", "5", "6"]
        msg.position = position

        if hand_type == "left":
            self.left_hand_publisher.publish(msg)
            self.left_hand_pos = position
            self.get_logger().debug(f"Left hand control command sent: {position}")
        else:
            self.right_hand_publisher.publish(msg)
            self.get_logger().debug(f"Right hand control command sent: {position}")

    def publish_action(self, action):
        """Publish action. Action vector layout (26-dim):
        [0:7] left arm, [7:13] left hand, [13:20] right arm, [20:26] right hand
        """
        target_joint = np.concatenate([action[:7], action[13:20]])
        left_hand_pos = action[7:13]
        right_hand_pos = action[20:]

        self.dual_arm_controller.publish(self._construct_dual_arm_ctrl_msg(target_joint))
        self.control_hand("left", left_hand_pos)
        self.control_hand("right", right_hand_pos)

    def get_obs(self):
        obs = {"images": {"camera_head": None}, "arm_gripper_joints": None}

        dual_arm_hand_status = self.get_current_preprospective()
        rgb, depth = self.get_current_imgs()
        if rgb is None:
            self.get_logger().warning("Images not ready yet")
            return None

        obs["images"]["camera_head"] = rgb
        obs["arm_gripper_joints"] = dual_arm_hand_status
        return obs

    def reset_home(self):
        state_3 = [
            -0.1525799448897199, 0.06799564128968774, 0.1352429110829423,
            -1.1551348918821753, 0.12439771977866568, -0.36139144432253956,
            -0.00591924481275605, -0.29126099842350656, -0.003778287841052544,
            -0.13665378849680831, -0.8683540414019328, -0.287210096964022,
            -0.4483082608478825, 0.19435190805574742,
        ]
        self.get_logger().info("Resetting to initial position...")
        time.sleep(3)
        self.reach_target_joint(state_3)

        self.control_hand("left", 1.0)
        self.control_hand("right", 1.0)

        self.get_logger().info("Reset to initial position completed")

    def warm_up(self):
        time.sleep(3)
        self.get_logger().info("System warm-up completed")

    def run(self):
        self.warm_up()
        self.get_logger().info("Starting reset home")
        self.reset_home()
        self.get_logger().info("Ending reset home")

        self.get_logger().info("Starting main loop")

        while rclpy.ok():
            obs = self.get_obs()
            if obs is None:
                self.get_logger().warning("Observations not ready yet")
                time.sleep(0.1)
                continue
            action = self.action_policy.inference(obs)
            self.get_logger().info(f"Policy output: {action[0].numpy()}")
            self.publish_action(action[0].numpy())
            time.sleep(0.05)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PolicyAgentNode()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)

        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()

        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
