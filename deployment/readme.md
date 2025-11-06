# ROS2 Deployment Guide

A VLA (Vision-Language-Action) deployment example for BrainCo dexterous hand integration with ROS2 system. 

## Steps

### 1. Model Setup
Edit the model path in `ros2_deployment_HAND.py`:
```python
model_path = "PATH_TO_YOUR_MODEL"
```

### 2. Start ROS2 Nodes
Launch the required ROS2 nodes for hardware communication.

### 3. Run Policy Inference
Execute the deployment script:
```bash
python ros2_deployment_HAND.py
```

## Files Description
- `ros2_deployment_HAND.py`: Main deployment script with PolicyAgentNode.
- `action_policy.py`: Policy inference wrapper class.

## Joint Value Order and publish_action Mapping

To ensure correct action dispatch, strictly follow this order when constructing/reading the 26‑dim `action` vector:
- indices `0..6`: left arm, 7 joints (`left_arm`)
- indices `7..12`: left hand, 6 joints (`left_hand`)
- indices `13..19`: right arm, 7 joints (`right_arm`)
- indices `20..25`: right hand, 6 joints (`right_hand`)

In `ros2_deployment_inspire.py`, `publish_action(self, action)` must slice according to the same order:
```python
def publish_action(self, action):
    target_joint = np.concatenate([action[:7], action[13:20]])  # left arm 7 + right arm 7
    left_hand_pos = action[7:13]                                # left hand 6
    right_hand_pos = action[20:26]                              # right hand 6
```

 