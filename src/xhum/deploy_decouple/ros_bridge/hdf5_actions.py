# -*- coding: utf-8 -*-
"""Load action trajectories for HDF5 replay (no ROS imports)."""

from __future__ import annotations

import h5py
import numpy as np


def load_action_trajectory(h5_path: str) -> list[np.ndarray]:
    """Return one action vector per timestep for ``publish_action``.

    Supported layouts:

    1. ``puppet/joint_position`` with shape ``(T, 26)`` — full command vector
       (Inspire / BrainCo 26-dim layout used by ``ros2_node_zmq.publish_action``).

    2. Legacy aligned groups (RoboMIND-style)::

         puppet/arm_left_position_align/data
         puppet/arm_right_position_align/data
         puppet/end_effector_left_position_align/data
         puppet/end_effector_right_position_align/data

       Concatenated per frame as
       ``[left_arm(7), left_hand(6), right_arm(7), right_hand(6)]``.
    """
    with h5py.File(h5_path, "r") as f:
        if "puppet/joint_position" in f:
            arr = np.asarray(f["puppet/joint_position"])
            if arr.ndim == 2 and arr.shape[1] == 26:
                return [np.asarray(row, dtype=np.float64) for row in arr]

        left_arm = np.asarray(f["puppet/arm_left_position_align/data"])
        right_arm = np.asarray(f["puppet/arm_right_position_align/data"])
        left_hand = np.asarray(f["puppet/end_effector_left_position_align/data"])
        right_hand = np.asarray(f["puppet/end_effector_right_position_align/data"])
        t = min(left_arm.shape[0], right_arm.shape[0], left_hand.shape[0], right_hand.shape[0])
        return [
            np.concatenate([left_arm[i], left_hand[i], right_arm[i], right_hand[i]]).astype(np.float64)
            for i in range(t)
        ]
