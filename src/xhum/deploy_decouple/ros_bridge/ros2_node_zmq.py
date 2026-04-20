#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 deploy node (Python 3.10) with remote policy over ZMQ.

Same robot I/O as ``src/xhum/deploy/ros2_deploy.py``; ``mode=model`` uses
``PolicyClient`` to talk to ``algorithm/policy_server.py`` (Python 3.12 +
LeRobot) instead of loading torch inside this process.

``mode=replay`` is **open-loop**: RGB/depth subscriptions are **not** created;
only arm/hand I/O and HDF5 actions are used (no camera topics required).

Run (after sourcing ROS):
  python3 ros2_node_zmq.py --config /path/to/config_zmq.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Tuple

# Allow running as a loose script (not installed as a package)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

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

from hdf5_actions import load_action_trajectory

# Arm command ROS topics (not configurable; remap at launch if needed)
_ARM_STATUS_TOPIC = "/arm/status"
_ARM_CMD_POS_TOPIC = "/arm/cmd_pos"
_ARM_FLEX_FREQ_TOPIC = "/joint_states_flex_freq"

_ARM_CMD_MODES = frozenset({"cmd_pos", "flex_freq"})
_HAND_TYPES = frozenset({"brainco", "inspire"})
_RUN_MODES = frozenset({"model", "replay"})

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
    "policy_server_url": "tcp://127.0.0.1:5555",
    "h5_path": "PATH_TO_H5",
    "camera_name": "camera",
    "action_rate": 20.0,
    # Only ``mode`` is read from YAML; arm ROS topics are fixed in this module.
    "arm_command": {"mode": "cmd_pos"},
}


def _normalize_arm_command(config: dict) -> dict:
    """Resolve arm publisher mode: ``arm_command.mode`` is ``cmd_pos`` or ``flex_freq``.

    Topics are constants ``_ARM_CMD_POS_TOPIC`` / ``_ARM_FLEX_FREQ_TOPIC``. Legacy
    ``arm_cmd_mode`` is accepted only when ``arm_command`` is absent from the file.
    """
    if "arm_flex_freq_topic" in config:
        raise ValueError(
            "Remove arm_flex_freq_topic from your YAML; arm topics are fixed in code "
            f"(flex_freq publishes JointState on {_ARM_FLEX_FREQ_TOPIC})."
        )
    default_mode = "cmd_pos"
    has_block = "arm_command" in config and config["arm_command"] is not None
    has_legacy_mode = "arm_cmd_mode" in config
    if has_block and has_legacy_mode:
        raise ValueError("Use only `arm_command.mode`, not `arm_cmd_mode`, in the same file.")
    if has_block:
        blk = config["arm_command"]
        if not isinstance(blk, dict):
            raise ValueError("arm_command must be a mapping")
        extra = set(blk.keys()) - {"mode"}
        if extra:
            raise ValueError(
                f"arm_command only supports key 'mode' (topics are fixed in code); remove: {sorted(extra)}"
            )
        if "mode" not in blk:
            raise ValueError(
                "arm_command must include the key 'mode' (expected 'cmd_pos' or 'flex_freq')."
            )
        mode = blk["mode"]
    elif "arm_cmd_mode" in config:
        mode = config["arm_cmd_mode"]
    else:
        mode = default_mode

    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(f"arm_command.mode must be a non-empty string, got {mode!r}")
    mode = mode.strip()
    if mode not in _ARM_CMD_MODES:
        raise ValueError(f"arm_command.mode must be one of {sorted(_ARM_CMD_MODES)}, got {mode!r}")
    return {"mode": mode}


def load_config(config_path: str | None, logger) -> dict:
    if config_path is None:
        config_path = str(_THIS_DIR / "config_zmq.example.yaml")
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

    merged["arm_command"] = _normalize_arm_command(config)
    merged.pop("arm_cmd_mode", None)
    return merged


def load_hdf5_actions(h5_path: str, logger) -> list[np.ndarray]:
    try:
        actions = load_action_trajectory(h5_path)
        logger.info(f"Loaded {len(actions)} actions from {h5_path}")
        return actions
    except Exception as e:
        logger.error(f"Failed to load HDF5 '{h5_path}': {e}")
        return []


class PolicyAgentNode(Node):
    def __init__(self, config_path: str | None = None):
        super().__init__("policy_agent_node_zmq")

        self.config = load_config(config_path, self.get_logger())
        if "mode" not in self.config:
            raise ValueError("config must contain key 'mode'")
        if "hand_type" not in self.config:
            raise ValueError("config must contain key 'hand_type'")
        if "arm_command" not in self.config or not isinstance(self.config["arm_command"], dict):
            raise ValueError("config must contain key 'arm_command' (mapping)")
        if "mode" not in self.config["arm_command"]:
            raise ValueError("arm_command must contain key 'mode'")

        self.mode = self.config["mode"]
        self.hand_type = self.config["hand_type"]

        if self.mode == "model":
            # Lazy import: replay-only runs do not need pyzmq / policy_client.
            from policy_client import PolicyClient

            url = self.config["policy_server_url"]
            self.action_policy = PolicyClient(server_url=url)
            self.get_logger().info(f"Mode=MODEL  policy_server={url}")
        elif self.mode == "replay":
            self.h5_path = self.config["h5_path"]
            self.action_policy = None
            self.get_logger().info(f"Mode=REPLAY  h5={self.h5_path}")
        else:
            raise ValueError(
                f"config.mode must be one of {sorted(_RUN_MODES)}, got {self.mode!r}"
            )

        self.left_hand_pos = np.zeros(6)
        self.right_hand_pos = np.zeros(6)

        if self.hand_type == "brainco":
            self._setup_brainco_hands()
        elif self.hand_type == "inspire":
            self._setup_inspire_hands()
        else:
            raise ValueError(
                f"config.hand_type must be one of {sorted(_HAND_TYPES)}, got {self.hand_type!r}"
            )

        self.joint_state_sub = self.create_subscription(MotorStatusMsg, _ARM_STATUS_TOPIC, self._arm_callback, 10)
        self.left_jpos = None
        self.right_jpos = None

        self._use_camera = self.mode == "model"
        self.bridge = None
        self.image = None
        self.depth = None
        self.rgb_sub = None
        self.depth_sub = None
        self.ats = None
        if self._use_camera:
            self.bridge = CvBridge()
            camera_name = self.config["camera_name"]
            self.rgb_sub = Subscriber(self, Image, f"/{camera_name}/color/image_raw")
            self.depth_sub = Subscriber(self, Image, f"/{camera_name}/depth/image_raw")
            self.ats = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
            self.ats.registerCallback(self._image_callback)
            self.get_logger().info(f"Camera sync: /{camera_name}/color|depth/image_raw (model mode)")
        else:
            self.get_logger().info("Mode=replay: no camera subscriptions (open-loop HDF5 replay)")

        ac = self.config["arm_command"]
        if "mode" not in ac:
            raise ValueError("arm_command must contain key 'mode'")
        self._arm_cmd_mode = ac["mode"]
        if self._arm_cmd_mode not in _ARM_CMD_MODES:
            raise ValueError(
                f"arm_command.mode must be one of {sorted(_ARM_CMD_MODES)}, got {self._arm_cmd_mode!r}"
            )
        self.dual_arm_controller = None
        self.arm_flex_freq_publisher = None
        if self._arm_cmd_mode == "cmd_pos":
            if not BODYCTRL_AVAILABLE:
                raise RuntimeError(
                    "arm_command.mode=cmd_pos requires bodyctrl_msgs "
                    "(CmdSetMotorPosition / SetMotorPosition on /arm/cmd_pos)."
                )
            self.dual_arm_controller = self.create_publisher(CmdSetMotorPosition, _ARM_CMD_POS_TOPIC, 10)
            self.get_logger().info(f"Arm command: CmdSetMotorPosition -> {_ARM_CMD_POS_TOPIC}")
        elif self._arm_cmd_mode == "flex_freq":
            self.arm_flex_freq_publisher = self.create_publisher(JointState, _ARM_FLEX_FREQ_TOPIC, 10)
            self.get_logger().info(f"Arm command: JointState (flex_freq) -> {_ARM_FLEX_FREQ_TOPIC}")
        else:
            raise RuntimeError(f"invalid arm_command.mode (internal): {self._arm_cmd_mode!r}")

        self.get_logger().info(f"PolicyAgentNode (ZMQ) init complete  hand_type={self.hand_type}")

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
        if not self._use_camera or self.bridge is None:
            return
        try:
            self.image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            self.depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def get_current_arm_status(self):
        if self.left_jpos is None or self.right_jpos is None:
            self.get_logger().warning("Arm status not ready")
            return np.zeros(14)
        return np.concatenate([self.left_jpos, self.right_jpos])

    def get_current_imgs(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.image, self.depth

    def get_current_hand_position(self, side: str = "left"):
        if side == "left":
            pos = self.left_hand_pos
        elif side == "right":
            pos = self.right_hand_pos
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        return np.asarray(pos).flatten()

    def get_current_proprioception(self):
        arm = self.get_current_arm_status()
        lh = self.get_current_hand_position("left")
        rh = self.get_current_hand_position("right")

        if self.hand_type == "brainco":
            return np.concatenate([arm, lh, rh])
        if self.hand_type == "inspire":
            return np.concatenate([arm[:7], lh, arm[7:], rh])
        raise ValueError(
            f"hand_type must be one of {sorted(_HAND_TYPES)}, got {self.hand_type!r}"
        )

    def _construct_dual_arm_ctrl_msg(self, target_joint: list[float]) -> CmdSetMotorPosition:
        """bodyctrl: one CmdSetMotorPosition wrapping 14× SetMotorPosition (motor id + pos/spd/cur)."""
        msg = CmdSetMotorPosition()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()

        arm_spd = float(self.config["arm_spd"])
        arm_cur = float(self.config["arm_cur"])

        for idx, val in enumerate(target_joint):
            cmd = SetMotorPosition()
            cmd.name = int(11 + idx if idx < 7 else 14 + idx)
            cmd.pos = float(np.asarray(val).reshape(()))
            cmd.spd = arm_spd
            cmd.cur = arm_cur
            msg.cmds.append(cmd)
        return msg

    def _construct_arm_flex_freq_msg(self, target_joint: list[float]) -> JointState:
        """sensor_msgs: JointState for flex-freq arm topic (names \"1\"..\"14\", rad, velocity zeros)."""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [str(i) for i in range(1, 15)]
        msg.position = [float(x) for x in target_joint]
        msg.velocity = [0.0] * 14
        msg.effort = []
        return msg

    def _publish_arm_target(self, target_joint) -> None:
        tj = np.asarray(target_joint, dtype=np.float64).flatten().tolist()
        if len(tj) != 14:
            raise ValueError(f"arm target must have 14 elements, got {len(tj)}")
        if self._arm_cmd_mode == "cmd_pos":
            out = self._construct_dual_arm_ctrl_msg(tj)
            self.dual_arm_controller.publish(out)
        elif self._arm_cmd_mode == "flex_freq":
            out = self._construct_arm_flex_freq_msg(tj)
            self.arm_flex_freq_publisher.publish(out)
        else:
            raise RuntimeError(
                f"arm_command.mode must be one of {sorted(_ARM_CMD_MODES)}, got {self._arm_cmd_mode!r}"
            )

    def reach_target_joint(self, target_joint) -> bool:
        fine_step = 500
        current_status = self.get_current_arm_status()
        step_array = np.linspace(current_status, target_joint, fine_step)

        self.get_logger().info("Moving arm to target position...")
        for stp in step_array:
            self._publish_arm_target(stp)
            time.sleep(1.0 / 400.0)

        self.get_logger().info("Arm reached target position")
        return True

    def control_hand(self, side: str, position):
        if side not in ("left", "right"):
            raise ValueError(f"control_hand side must be 'left' or 'right', got {side!r}")
        if self.hand_type == "brainco":
            self._control_hand_brainco(side, position)
        elif self.hand_type == "inspire":
            self._control_hand_inspire(side, position)
        else:
            raise ValueError(
                f"hand_type must be one of {sorted(_HAND_TYPES)}, got {self.hand_type!r}"
            )

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
        elif side == "right":
            self.right_hand_publisher.publish(msg)
            self.right_hand_pos = np.asarray(msg.positions, dtype=np.float64)
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

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
        elif side == "right":
            self.right_hand_publisher.publish(msg)
            self.right_hand_pos = np.array(position)
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    def publish_action(self, action):
        if self.hand_type == "brainco":
            target_joint = action[:14]
            left_hand = action[14:20]
            right_hand = action[20:]
        elif self.hand_type == "inspire":
            target_joint = np.concatenate([action[:7], action[13:20]])
            left_hand = action[7:13]
            right_hand = action[20:]
        else:
            raise ValueError(
                f"hand_type must be one of {sorted(_HAND_TYPES)}, got {self.hand_type!r}"
            )

        self._publish_arm_target(target_joint)
        self.control_hand("left", left_hand)
        self.control_hand("right", right_hand)

    def get_obs(self):
        if not self._use_camera:
            return None
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

    def reset_home(self):
        home_pos = self.config["home_position"]
        home_wait = self.config["home_wait"]

        self.get_logger().info("Resetting to home position...")
        time.sleep(home_wait)
        self.reach_target_joint(home_pos)

        if self.hand_type == "brainco":
            self.control_hand("left", [99] * 6)
            self.control_hand("right", [99] * 6)
        elif self.hand_type == "inspire":
            self.control_hand("left", 1.0)
            self.control_hand("right", 1.0)
        else:
            raise ValueError(
                f"hand_type must be one of {sorted(_HAND_TYPES)}, got {self.hand_type!r}"
            )

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
            self.get_logger().info("Starting remote model inference loop")
            while rclpy.ok():
                obs = self.get_obs()
                if obs is None:
                    time.sleep(0.1)
                    continue
                action = self.action_policy.inference(obs)
                row = action[0]
                self.publish_action(np.asarray(row, dtype=np.float64))
                time.sleep(action_period)
        else:
            raise RuntimeError(f"run() invalid mode (internal): {self.mode!r}")


def main(args=None):
    parser = argparse.ArgumentParser(description="ROS2 deploy node (ZMQ remote policy)")
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
