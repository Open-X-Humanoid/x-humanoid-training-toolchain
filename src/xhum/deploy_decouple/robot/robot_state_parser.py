# -*- coding: utf-8 -*-
"""Parse ``ros2_bridge_msgs/RobotState`` into numpy arrays for deploy."""

from __future__ import annotations

from typing import Any

import numpy as np

# TienKung / evt: left 11–17, right 21–27 (rad)
ARM_MOTOR_IDS_LEFT = (11, 12, 13, 14, 15, 16, 17)
ARM_MOTOR_IDS_RIGHT = (21, 22, 23, 24, 25, 26, 27)
ARM_MOTOR_IDS = ARM_MOTOR_IDS_LEFT + ARM_MOTOR_IDS_RIGHT


def _motor_table(status_list: Any) -> dict[int, Any]:
    """``name -> MotorStatus`` from ``ArmStatus.status``."""
    out: dict[int, Any] = {}
    for m in status_list or []:
        out[int(m.name)] = m
    return out


def _ordered_values(
    table: dict[int, Any],
    motor_ids: tuple[int, ...],
    field: str,
) -> np.ndarray:
    missing = [mid for mid in motor_ids if mid not in table]
    if missing:
        raise KeyError(f"arm status missing motor id(s): {missing}")
    return np.array([float(getattr(table[mid], field)) for mid in motor_ids], dtype=np.float64)


def parse_arm_from_robot_state(msg: Any) -> dict[str, np.ndarray]:
    """Extract arm joint data from ``RobotState``.

    Returns dict with keys:
      ``left_pos``, ``right_pos``, ``left_speed``, ``right_speed``,
      ``left_current``, ``right_current``, ``positions`` (14,),
      ``speeds``, ``currents``.
    """
    if msg is None or not hasattr(msg, "arm") or msg.arm is None:
        raise ValueError("RobotState.arm is missing")

    table = _motor_table(msg.arm.status)

    left_pos = _ordered_values(table, ARM_MOTOR_IDS_LEFT, "pos")
    right_pos = _ordered_values(table, ARM_MOTOR_IDS_RIGHT, "pos")
    left_speed = _ordered_values(table, ARM_MOTOR_IDS_LEFT, "speed")
    right_speed = _ordered_values(table, ARM_MOTOR_IDS_RIGHT, "speed")
    left_current = _ordered_values(table, ARM_MOTOR_IDS_LEFT, "current")
    right_current = _ordered_values(table, ARM_MOTOR_IDS_RIGHT, "current")

    return {
        "left_pos": left_pos,
        "right_pos": right_pos,
        "left_speed": left_speed,
        "right_speed": right_speed,
        "left_current": left_current,
        "right_current": right_current,
        "positions": np.concatenate([left_pos, right_pos]),
        "speeds": np.concatenate([left_speed, right_speed]),
        "currents": np.concatenate([left_current, right_current]),
    }


def parse_arm_positions(msg: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(left_arm_7, right_arm_7)`` joint positions in rad."""
    parsed = parse_arm_from_robot_state(msg)
    return parsed["left_pos"], parsed["right_pos"]


def parse_arm_positions_flat(msg: Any) -> np.ndarray:
    """Return 14-dim vector: left 7 + right 7 (same order as ``ARM_MOTOR_IDS``)."""
    return parse_arm_from_robot_state(msg)["positions"]
