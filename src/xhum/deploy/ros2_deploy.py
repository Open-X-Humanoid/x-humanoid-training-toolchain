#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified ROS2 deployment node for TienKung robot.

Supports BrainCo and Inspire dexterous hands via `hand_type` config,
and model inference / HDF5 replay via `mode` config.

Usage:
    xhum-deploy --config my_config.yaml
"""

import argparse
import os
import threading
import time
from typing import Tuple

import h5py
import numpy as np
import rclpy
import yaml
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
    print("Warning: bodyctrl_msgs not available, using String as fallback")

try:
    from ros2_stark_interfaces.msg import MotorStatus, SetMotorMulti

    BRAINCO_AVAILABLE = True
except ImportError:
    BRAINCO_AVAILABLE = False

from xhum.deploy.policy_agent import PolicyAgent

# ──────────────────────────────────────────────────────────────────────
# Per-hand-type defaults
# ──────────────────────────────────────────────────────────────────────

_BRAINCO_HOME = [
    -0.05916397, 0.11694484, 0.00816471, -1.6296118, -0.18107964, -0.1322771, -0.08812793,
    -0.00609963, 0.05809595, -0.0326848, -1.6615903, -0.15082923, 0.03735191, 0.00886455,
]

_INSPIRE_HOME = [
    -0.1525799448897199, 0.06799564128968774, 0.1352429110829423,
    -1.1551348918821753, 0.12439771977866568, -0.36139144432253956,
    -0.00591924481275605, -0.29126099842350656, -0.003778287841052544,
    -0.13665378849680831, -0.8683540414019328, -0.287210096964022,
    -0.4483082608478825, 0.19435190805574742,
]

HAND_TYPE_DEFAULTS = {
    "brainco": {
        "arm_spd": 150.0,
        "arm_cur": 80.0,
        "obs_camera_key": "camera",
        "home_position": _BRAINCO_HOME,
        "home_wait": 5,
    },
    "inspire": {
        "arm_spd": 0.5,
        "arm_cur": 5.0,
        "obs_camera_key": "camera_head",
        "home_position": _INSPIRE_HOME,
        "home_wait": 3,
    },
}

DEFAULT_CONFIG = {
    "mode": "model",
    "hand_type": "inspire",
    "model_path": "PATH_TO_MODEL",
    "h5_path": "PATH_TO_H5",
    "camera_name": "camera",
    "action_rate": 20.0,
}

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def load_config(config_path: str | None, logger) -> dict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Configuration loaded from: {config_path}")
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {}
    except Exception as e:
        logger.error(f"Error loading config: {e}, using defaults")
        config = {}

    merged = {**DEFAULT_CONFIG, **config}
    hand_defaults = HAND_TYPE_DEFAULTS.get(merged["hand_type"], {})
    for k, v in hand_defaults.items():
        merged.setdefault(k, v)
    return merged


def load_hdf5_actions(h5_path: str, logger) -> list[np.ndarray]:
    """Load action sequence from HDF5.

    Per-frame layout: [left_arm(7), left_hand(6), right_arm(7), right_hand(6)]
    """
    try:
        with h5py.File(h5_path, "r") as f:
            left_arm = np.asarray(f["puppet/arm_left_position_align/data"])
            right_arm = np.asarray(f["puppet/arm_right_position_align/data"])
            left_hand = np.asarray(f["puppet/end_effector_left_position_align/data"])
            right_hand = np.asarray(f["puppet/end_effector_right_position_align/data"])
        T = min(left_arm.shape[0], right_arm.shape[0], left_hand.shape[0], right_hand.shape[0])
        actions = [
            np.concatenate([left_arm[t], left_hand[t], right_arm[t], right_hand[t]]).astype(np.float64)
            for t in range(T)
        ]
        logger.info(f"Loaded {len(actions)} actions from {h5_path}")
        return actions
    except Exception as e:
        logger.error(f"Failed to load HDF5 '{h5_path}': {e}")
        return []


# ──────────────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────────────


class PolicyAgentNode(Node):
    def __init__(self, config_path: str | None = None):
        super().__init__("policy_agent_node")

        self.config = load_config(config_path, self.get_logger())
        self.mode = self.config["mode"]
        self.hand_type = self.config["hand_type"]

        # ── mode ──
        if self.mode == "model":
            self.action_policy = PolicyAgent(self.config["model_path"])
            self.get_logger().info(f"Mode=MODEL  model={self.config['model_path']}")
        elif self.mode == "replay":
            self.h5_path = self.config["h5_path"]
            self.action_policy = None
            self.get_logger().info(f"Mode=REPLAY  h5={self.h5_path}")
        else:
            raise ValueError(f"Unknown mode '{self.mode}', must be 'model' or 'replay'")

        # ── hand-specific pub / sub ──
        self.left_hand_pos = np.zeros(6)
        self.right_hand_pos = np.zeros(6)

        if self.hand_type == "brainco":
            self._setup_brainco_hands()
        elif self.hand_type == "inspire":
            self._setup_inspire_hands()
        else:
            raise ValueError(f"Unknown hand_type '{self.hand_type}', must be 'brainco' or 'inspire'")

        # ── arm ──
        self.joint_state_sub = self.create_subscription(MotorStatusMsg, "/arm/status", self._arm_callback, 10)
        self.left_jpos = None
        self.right_jpos = None

        # ── camera ──
        self.bridge = CvBridge()
        self.image = None
        self.depth = None

        camera_name = self.config["camera_name"]
        self.rgb_sub = Subscriber(self, Image, f"/{camera_name}/color/image_raw")
        self.depth_sub = Subscriber(self, Image, f"/{camera_name}/depth/image_raw")
        self.ats = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ats.registerCallback(self._image_callback)

        # ── arm control publisher ──
        self.dual_arm_controller = self.create_publisher(CmdSetMotorPosition, "/arm/cmd_pos", 10)

        self.get_logger().info(f"PolicyAgentNode init complete  hand_type={self.hand_type}")

    # ================================================================
    # Hand setup (BrainCo / Inspire)
    # ================================================================

    def _setup_brainco_hands(self):
        if not BRAINCO_AVAILABLE:
            raise ImportError("ros2_stark_interfaces is required for hand_type='brainco'")

        self.left_hand_publisher = self.create_publisher(SetMotorMulti, "/left_hand/set_motor_multi", 10)
        self.right_hand_publisher = self.create_publisher(SetMotorMulti, "/right_hand/set_motor_multi", 10)

        self.create_subscription(MotorStatus, "/left_hand/motor_status", self._brainco_left_cb, 10)
        self.create_subscription(MotorStatus, "/right_hand/motor_status", self._brainco_right_cb, 10)

    def _setup_inspire_hands(self):
        self.left_hand_publisher = self.create_publisher(JointState, "/inspire_hand/ctrl/left_hand", 10)
        self.right_hand_publisher = self.create_publisher(JointState, "/inspire_hand/ctrl/right_hand", 10)

        self.create_subscription(JointState, "/inspire_hand/state/left_hand", self._inspire_left_cb, 10)
        self.create_subscription(JointState, "/inspire_hand/state/right_hand", self._inspire_right_cb, 10)

    # ================================================================
    # Callbacks
    # ================================================================

    def _brainco_left_cb(self, msg):
        self.left_hand_pos = np.array(msg.positions)

    def _brainco_right_cb(self, msg):
        self.right_hand_pos = np.array(msg.positions)

    def _inspire_left_cb(self, msg):
        if len(msg.position) > 0:
            self.left_hand_pos = np.array(msg.position)

    def _inspire_right_cb(self, msg):
        if len(msg.position) > 0:
            self.right_hand_pos = np.array(msg.position)

    def _arm_callback(self, msg):
        tmp = [val.pos for val in msg.status]
        self.left_jpos = tmp[:7]
        self.right_jpos = tmp[7:]

    def _image_callback(self, rgb_msg, depth_msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            self.depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    # ================================================================
    # State queries
    # ================================================================

    def get_current_arm_status(self):
        if self.left_jpos is None or self.right_jpos is None:
            self.get_logger().warning("Arm status not ready")
            return np.zeros(14)
        return np.concatenate([self.left_jpos, self.right_jpos])

    def get_current_imgs(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.image, self.depth

    def get_current_hand_position(self, side: str = "left"):
        pos = self.left_hand_pos if side == "left" else self.right_hand_pos
        return np.asarray(pos).flatten()

    def get_current_proprioception(self):
        """Build proprioception vector. Layout depends on hand_type:
        - brainco: [arm(14), left_hand(6), right_hand(6)]
        - inspire: [arm_left(7), left_hand(6), arm_right(7), right_hand(6)]
        """
        arm = self.get_current_arm_status()
        lh = self.get_current_hand_position("left")
        rh = self.get_current_hand_position("right")

        if self.hand_type == "brainco":
            return np.concatenate([arm, lh, rh])
        else:
            return np.concatenate([arm[:7], lh, arm[7:], rh])

    # ================================================================
    # Control helpers
    # ================================================================

    def _construct_dual_arm_ctrl_msg(self, target_joint: list[float]):
        msg = CmdSetMotorPosition()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()

        arm_spd = self.config["arm_spd"]
        arm_cur = self.config["arm_cur"]

        for idx, val in enumerate(target_joint):
            cmd = SetMotorPosition()
            cmd.name = 11 + idx if idx < 7 else 14 + idx
            cmd.pos = val.item()
            cmd.spd = arm_spd
            cmd.cur = arm_cur
            msg.cmds.append(cmd)
        return msg

    def reach_target_joint(self, target_joint) -> bool:
        fine_step = 500
        current_status = self.get_current_arm_status()
        step_array = np.linspace(current_status, target_joint, fine_step)

        self.get_logger().info("Moving arm to target position...")
        for stp in step_array:
            self.dual_arm_controller.publish(self._construct_dual_arm_ctrl_msg(stp))
            time.sleep(1.0 / 400.0)

        self.get_logger().info("Arm reached target position")
        return True

    def control_hand(self, side: str, position):
        """Send hand control command. Dispatches to hand-type-specific logic."""
        if self.hand_type == "brainco":
            self._control_hand_brainco(side, position)
        else:
            self._control_hand_inspire(side, position)

    def _control_hand_brainco(self, side: str, position):
        msg = SetMotorMulti()
        if isinstance(position, (list, np.ndarray)):
            msg.positions = np.asarray(position, dtype=np.uint16)
        else:
            msg.positions = np.array([int(position)] * 6, dtype=np.uint16)
        msg.mode = 1

        if side == "left":
            self.left_hand_publisher.publish(msg)
            self.left_hand_pos = np.asarray(msg.positions, dtype=np.float64)
        else:
            self.right_hand_publisher.publish(msg)
            self.right_hand_pos = np.asarray(msg.positions, dtype=np.float64)

    def _control_hand_inspire(self, side: str, position):
        if isinstance(position, (list, np.ndarray)):
            position = [round(np.clip(float(p), 0, 1), 1) for p in position]
        else:
            position = [round(np.clip(float(position), 0, 1), 1)] * 6

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["1", "2", "3", "4", "5", "6"]
        msg.position = position

        if side == "left":
            self.left_hand_publisher.publish(msg)
            self.left_hand_pos = np.array(position)
        else:
            self.right_hand_publisher.publish(msg)
            self.right_hand_pos = np.array(position)

    def publish_action(self, action):
        """Publish action to arm + hands. Layout depends on hand_type:
        - brainco: [arm(14), left_hand(6), right_hand(6)]
        - inspire: [arm_left(7), left_hand(6), arm_right(7), right_hand(6)]
        """
        if self.hand_type == "brainco":
            target_joint = action[:14]
            left_hand = action[14:20]
            right_hand = action[20:]
        else:
            target_joint = np.concatenate([action[:7], action[13:20]])
            left_hand = action[7:13]
            right_hand = action[20:]

        self.dual_arm_controller.publish(self._construct_dual_arm_ctrl_msg(target_joint))
        self.control_hand("left", left_hand)
        self.control_hand("right", right_hand)

    def get_obs(self):
        proprioception = self.get_current_proprioception()
        rgb, _depth = self.get_current_imgs()
        if rgb is None:
            self.get_logger().warning("Images not ready")
            return None

        cam_key = self.config["obs_camera_key"]
        return {
            "images": {cam_key: rgb},
            "arm_gripper_joints": proprioception,
        }

    # ================================================================
    # Reset & run
    # ================================================================

    def reset_home(self):
        home_pos = self.config["home_position"]
        home_wait = self.config["home_wait"]

        self.get_logger().info("Resetting to home position...")
        time.sleep(home_wait)
        self.reach_target_joint(home_pos)

        if self.hand_type == "brainco":
            self.control_hand("left", [99] * 6)
            self.control_hand("right", [99] * 6)
        else:
            self.control_hand("left", 1.0)
            self.control_hand("right", 1.0)

        self.get_logger().info("Home position reached")

    def warm_up(self):
        time.sleep(3)
        self.get_logger().info("Warm-up completed")

    def run(self):
        self.warm_up()
        self.reset_home()

        action_rate = self.config.get("action_rate", 20.0)
        action_period = 1.0 / action_rate

        if self.mode == "replay":
            actions = load_hdf5_actions(self.h5_path, self.get_logger())
            if not actions:
                self.get_logger().error(f"No actions loaded from {self.h5_path}")
                return

            self.get_logger().info(f"Streaming {len(actions)} HDF5 actions...")
            for act in actions:
                self.publish_action(act)
                time.sleep(action_period)
            self.get_logger().info("Finished streaming HDF5 actions.")

        elif self.mode == "model":
            self.get_logger().info("Starting model inference loop")
            while rclpy.ok():
                obs = self.get_obs()
                if obs is None:
                    time.sleep(0.1)
                    continue
                action = self.action_policy.inference(obs)
                self.publish_action(action[0].numpy())
                time.sleep(action_period)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────


def main(args=None):
    parser = argparse.ArgumentParser(description="Unified ROS2 deployment node")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args_parsed, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    try:
        node = PolicyAgentNode(config_path=args_parsed.config)
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
