#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from ros2_stark_interfaces.msg import MotorStatus, SetMotorMulti
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# Import message types
try:
    from bodyctrl_msgs.msg import CmdSetMotorPosition, MotorStatusMsg, SetMotorPosition
    BODYCTRL_AVAILABLE = True
except ImportError:
    from std_msgs.msg import String as CmdSetMotorPosition
    from std_msgs.msg import String as MotorStatusMsg
    from std_msgs.msg import String as SetMotorPosition
    BODYCTRL_AVAILABLE = False
    print("Warning: bodyctrl_msgs package is not available, will use String message type as replacement")

# Use message_filters for message synchronization
from action_policy import PolicyAgent
from message_filters import ApproximateTimeSynchronizer, Subscriber


class PolicyAgentNode(Node):
    def __init__(self):
        # Initialize node
        super().__init__('policy_agent_node')
        
        # DEBUG
        self.cnt = 0
        
        #reply
        self.left_hand_joints_list, self.right_hand_joints_list, self.arm_status_list = [], [], []
        self.reply = False
        # model_path = "CERTAIN_TASK/checkpoints/200000/pretrained_model"
        model_path = "PATH_TO_MODEL"
        if not self.reply:
            self.action_policy = PolicyAgent(model_path)
        
        # Initialize hand controller
        # In ROS2, we use JointState messages to control hands instead of direct serial port
        self.left_hand_publisher = self.create_publisher(
            SetMotorMulti,
            '/left_hand/set_motor_multi',
            10)
            
        # Create right hand control publisher
        self.right_hand_publisher = self.create_publisher(
            SetMotorMulti,
            '/right_hand/set_motor_multi',
            10)
            
        #TODO brainco
        # Subscribe to left hand status
        self.left_hand_brainco_subscription = self.create_subscription(
            MotorStatus,
            '/left_hand/motor_status',
            self.left_hand_brainco_callback,
            10)
        
        # Subscribe to right hand status
        self.right_hand_brainco_subscription = self.create_subscription(
            MotorStatus,
            '/right_hand/motor_status',
            self.right_hand_brainco_callback,
            10)
            
        # Current joint positions for left and right hands
        self.left_hand_pos = 0.0
        self.right_hand_pos = 0.0
        
        # Subscribe to robotic arm status
        self.joint_state_sub = self.create_subscription(
            MotorStatusMsg,
            '/arm/status',
            self.arm_callback,
            10)
            
        self.left_jpos = None
        self.right_jpos = None
        
        # Create CV Bridge for converting between ROS image messages and OpenCV images
        self.bridge = CvBridge()
        self.image = None
        self.depth = None
        
        # Set camera name
        camera_name = 'camera'
        
        # Use message_filters for message synchronization
        self.rgb_sub = Subscriber(self, Image, f'/{camera_name}/color/image_raw')
        self.depth_sub = Subscriber(self, Image, f'/{camera_name}/depth/image_raw')
        
        # Create synchronizer
        self.ats = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ats.registerCallback(self.image_callback)
        
        # Create robotic arm control publisher
        self.dual_arm_controller = self.create_publisher(
            CmdSetMotorPosition,
            '/arm/cmd_pos',
            10)
            
        self.get_logger().info('PolicyAgentNode initialization completed')
    
    def left_hand_brainco_callback(self, msg):
        # Process left hand status message

        self.left_hand_pos = msg.positions
        self.get_logger().debug(f'Received left hand status: {self.left_hand_pos}')
    
    def right_hand_brainco_callback(self, msg):
        # Process right hand status message
        self.right_hand_pos = msg.positions
        self.get_logger().debug(f'Received left hand status: {self.right_hand_pos}')
    

    def arm_callback(self, msg):
        tmp_arms_status = []
        for val in msg.status:
            tmp_arms_status.append(val.pos)
        self.left_jpos = tmp_arms_status[:7]
        self.right_jpos = tmp_arms_status[7:]
    
    def warm_up(self):
        # Wait for a while to allow subscribers to receive messages
        time.sleep(3)
        self.get_logger().info('System warm-up completed')

    def get_current_arm_status(self):
        # Get current robotic arm status
        if self.left_jpos is None or self.right_jpos is None:
            self.get_logger().warning('Robotic arm status not ready yet')
            return np.zeros(14)  # Assume 14 joints in total
        
        self.get_logger().debug(f'Left arm: {self.left_jpos}')
        self.get_logger().debug(f'Right arm: {self.right_jpos}')
        return np.concatenate([self.left_jpos, self.right_jpos])
    
    def image_callback(self, rgb_msg, depth_msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            self.depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            self.get_logger().debug('Synchronized color and depth images received')
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
   
    def get_current_imgs(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.image, self.depth

    def get_current_hand_position(self, hand_type='left'):
        # Get current hand position
        if hand_type == 'left':
            return np.array(self.left_hand_pos)  # Return 1D array
        else:  # right
            return np.array(self.right_hand_pos)  # Return 1D array

    def get_current_preprospective(self):
        # Get current status of robotic arm and hands
        arm_status = self.get_current_arm_status()
        left_hand = self.get_current_hand_position('left').flatten()  # Ensure 1D array
        right_hand = self.get_current_hand_position('right').flatten()  # Ensure 1D array
        #TODO
        # left_hand = [0] #self.get_current_hand_position('left').flatten()  # Ensure 1D array
        # right_hand = [1] #self.get_current_hand_position('right').flatten()  # Ensure 1D array
        return np.concatenate([arm_status, left_hand, right_hand])

    def _construct_dual_arm_ctrl_msg(self, target_joint: list[float]):
        msg = CmdSetMotorPosition()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        print("target_joint",target_joint)
        for idx, val in enumerate(target_joint):
            cmd = SetMotorPosition()
            if idx < 7:
                cmd.name = 11+idx
            else:
                cmd.name = 14+idx 
            # cmd.pos = val.item()
            print("val",val)
            # cmd.pos = float(val)
            cmd.pos = val.item()
            cmd.spd = 150.0
            cmd.cur = 80.0
            msg.cmds.append(cmd)
        
        return msg
    
    def reach_target_joint(self, target_joint, asynchronous: bool = False) -> bool:
        fine_step = 500
        current_status = self.get_current_arm_status()
        step_array = np.linspace(current_status, target_joint, fine_step)
        
        self.get_logger().info('Start moving robotic arm to target position')
        for stp in step_array:
            self.dual_arm_controller.publish(self._construct_dual_arm_ctrl_msg(stp))
            # Use Python's time.sleep instead of ROS1's rate.sleep
            time.sleep(1.0/400.0)  # Equivalent to 400Hz frequency
        
        self.get_logger().info('Robotic arm has reached target position')
        return True
    
    # def control_hand(self, hand_type, position):
    #     # Control hand
    #     # Check if position is an array
    #     if isinstance(position, (list, np.ndarray)):
    #         # If it's an array, ensure each value is in 0-1 range
    #         position = [np.clip(float(pos), 0, 1) for pos in position]
    #         position = [round(pos, 1) for pos in position]
    #     else:
    #         # If it's a scalar, convert to array
    #         position = np.clip(float(position), 0, 1)
    #         position = round(position, 1)
    #         position = [position] * 6  # Set all 6 joints to the same position
        
    #     msg = JointState()
    #     msg.header.stamp = self.get_clock().now().to_msg()
        
    #     # Set joint names and positions
    #     msg.name = ['1', '2', '3', '4', '5', '6']
    #     # Use position array as joint positions
    #     msg.position = position
        
    #     if hand_type == 'left':
    #         self.left_hand_publisher.publish(msg)
    #         self.left_hand_pos = position
    #         self.get_logger().debug(f'Left hand control command sent: {position}')
    #     else:  # right
    #         self.right_hand_publisher.publish(msg)
    #         self.get_logger().debug(f'Right hand control command sent: {position}')

    #BrainCo
    def control_hand_brainco(self, hand_type, position):
        # Control hand
        # Check if position is an array
        
        msg = SetMotorMulti()

  
        print("position",position) 

        if position is not None:
            msg.positions = position.astype(np.uint16)

        msg.mode = 1
        # if velocities is not None:
        #     msg.velocity = velocities
        # if efforts is not None:
        #     msg.effort = efforts
        
        # 发布消息
        # print("msg",msg)
        if hand_type == 'left':
            self.left_hand_publisher.publish(msg)
            self.left_hand_pos = position
            self.get_logger().debug(f'Left hand control command sent: {position}')
        else:  # right
            self.right_hand_publisher.publish(msg)
            self.get_logger().debug(f'Right hand control command sent: {position}')
      
    def publish_action(self, action):
        # Publish action
        target_joint = action[:14]
        left_hand_pos = action[14:20]
        right_hand_pos = action[20:]
        
        self.get_logger().info(f"Target joints: {target_joint}")
        self.get_logger().info(f"Left hand position: {left_hand_pos}")
        self.get_logger().info(f"Right hand position: {right_hand_pos}")
        
        # Publish robotic arm control command
        self.dual_arm_controller.publish(self._construct_dual_arm_ctrl_msg(target_joint))
        # print("left_hand_pos",left_hand_pos)
        # Control left and right hands
        self.control_hand_brainco('left', left_hand_pos)
        self.control_hand_brainco('right', right_hand_pos)
    
    def get_obs(self):
        # Get observations
        obs = {
            'images': {
                'camera': None
            },
            'arm_gripper_joints': None
        }
        
        # Get robotic arm and hand status
        dual_arm_hand_status = self.get_current_preprospective()
        
        # Get images
        rgb, depth = self.get_current_imgs()
        if rgb is None:
            self.get_logger().warning("Images not ready yet")
            return None
        
        obs['images']['camera'] = rgb
        obs['arm_gripper_joints'] = dual_arm_hand_status
        return obs
    
    def get_obs_reply(self,count=0):
        # Get observations
        obs = {
            'images': {
                'camera': None
            },
            'arm_gripper_joints': None
        }
        
        # Get robotic arm and hand status
        dual_arm_hand_status = np.concatenate([self.arm_status_list[count], self.left_hand_joints_list[count], self.right_hand_joints_list[count]])
        
        
        obs['images']['camera'] = None
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


    def run_obs(self):
        self.warm_up()
        self.reset_home()
        
        self.get_logger().info("Starting main loop")
    
        print("Starting main loop")


        while rclpy.ok():
            # Get observations
            obs = self.get_obs() 
            # print("obs",obs)
            if obs is None:
                self.get_logger().warning("Observations not ready yet")
                time.sleep(0.1)  # Brief wait before retry
                continue
            # breakpoint()
            action = self.action_policy.inference(obs)
            self.get_logger().info(f"Policy output: {action[0][0].numpy()}")
            self.publish_action(action[0][0].numpy()) 
            time.sleep(0.05)

    def run(self):
        """Run the main loop of the node"""
        self.run_obs()
        


def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    try:
        # Create node
        node = PolicyAgentNode()
        
        # Use multi-threaded executor
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        
        # Create a thread to run the executor
        import threading
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        
        # Run main loop
        node.run() 
        
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        rclpy.shutdown()


if __name__ == '__main__':
    main()
