# -*- coding: utf-8 -*-
"""Load action trajectories for HDF5 replay (no ROS imports)."""

from __future__ import annotations

import h5py
import numpy as np


_LEGACY_KEYS = (
    "puppet/arm_left_position_align/data",
    "puppet/arm_right_position_align/data",
    "puppet/end_effector_left_position_align/data",
    "puppet/end_effector_right_position_align/data",
)


def load_action_trajectory(h5_path: str) -> list[np.ndarray]:
    """Return one 26-dim action vector per timestep for ``publish_action``.

    Two accepted layouts:

    1. ``puppet/joint_position`` with shape ``(T, 26)`` — full command vector
       (Inspire / BrainCo 26-dim layout used by ``ros2_node.publish_action``).

    2. Legacy aligned groups (RoboMIND-style), concatenated per frame as
       ``[left_arm(7), left_hand(6), right_arm(7), right_hand(6)]``::

         puppet/arm_left_position_align/data
         puppet/arm_right_position_align/data
         puppet/end_effector_left_position_align/data
         puppet/end_effector_right_position_align/data

    Raises a clear error when ``puppet/joint_position`` exists but has an
    unexpected shape (previously this silently fell through to legacy keys,
    masking the real cause).
    """
    with h5py.File(h5_path, "r") as f:
        if "puppet/joint_position" in f:
            arr = np.asarray(f["puppet/joint_position"])
            if arr.ndim != 2 or arr.shape[1] != 26:
                raise ValueError(
                    f"puppet/joint_position has shape {arr.shape}; expected (T, 26). "
                    "Fix the dataset or remove it if you meant to use the legacy *_align groups."
                )
            return [np.asarray(row, dtype=np.float64) for row in arr]

        missing = [k for k in _LEGACY_KEYS if k not in f]
        if missing:
            raise KeyError(
                "No action layout found: puppet/joint_position is absent and the legacy "
                f"aligned groups are incomplete (missing: {missing})."
            )

        left_arm = np.asarray(f["puppet/arm_left_position_align/data"])
        right_arm = np.asarray(f["puppet/arm_right_position_align/data"])
        left_hand = np.asarray(f["puppet/end_effector_left_position_align/data"])
        right_hand = np.asarray(f["puppet/end_effector_right_position_align/data"])
        t = min(left_arm.shape[0], right_arm.shape[0], left_hand.shape[0], right_hand.shape[0])
        return [
            np.concatenate(
                [left_arm[i], left_hand[i], right_arm[i], right_hand[i]]
            ).astype(np.float64)
            for i in range(t)
        ]
