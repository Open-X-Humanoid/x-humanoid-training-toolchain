# -*- coding: utf-8 -*-
"""YAML config loading + PolicyClient factory (no ROS2 imports).

Imported by ``ros2_node.py`` and the headless replay_debug path. Keeps
the ROS node thin: everything YAML-shaped lives here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


# Single source of truth for the canonical string sets and default home poses.
# ``ros2_node.py`` imports these — do not copy elsewhere.
ARM_CMD_MODES = frozenset({"cmd_pos", "flex_freq"})
HAND_TYPES = frozenset({"brainco", "inspire"})
RUN_MODES = frozenset({"model", "replay", "replay_actions", "replay_debug"})

_BRAINCO_HOME = [
    -0.05916397, 0.11694484, 0.00816471, -1.6296118, -0.18107964, -0.1322771, -0.08812793,
    -0.00609963, 0.05809595, -0.0326848, -1.6615903, -0.15057923, 0.03735191, 0.00886455,
]

_INSPIRE_HOME = [
    -0.1525799448897199, 0.06799564128968774, 0.1352429110829423,
    -1.1551348918821753, 0.12439771977866568, -0.36139144432253956,
    -0.00591924481275605, -0.29126099842350656, -0.003778287841052544,
    -0.13665378849680831, -0.8683540414019328, -0.287210096964022,
    -0.4483082608478825, 0.19435190805574742,
]

# Hand home defaults — mirror the values previously hard-coded in ros2_node.reset_home.
# Format matches what ``PolicyAgentNode.control_hand`` already accepts:
#   brainco: int list of 6 in [0..100] (99 ≈ fully open), or single int broadcast
#   inspire: float in [0..1] (1.0 ≈ fully open), or list of 6 floats
_BRAINCO_HAND_HOME = [99] * 6
_INSPIRE_HAND_HOME = 1.0

HAND_TYPE_DEFAULTS = {
    "brainco": {
        "arm_spd": 150.0,
        "arm_cur": 80.0,
        "obs_camera_key": "camera",
        "home_position": _BRAINCO_HOME,
        "home_wait": 5,
        "home_hand": {"left": list(_BRAINCO_HAND_HOME), "right": list(_BRAINCO_HAND_HOME)},
    },
    "inspire": {
        "arm_spd": 0.5,
        "arm_cur": 5.0,
        "obs_camera_key": "camera_head",
        "home_position": _INSPIRE_HOME,
        "home_wait": 3,
        "home_hand": {"left": _INSPIRE_HAND_HOME, "right": _INSPIRE_HAND_HOME},
    },
}

DEFAULT_CONFIG = {
    "mode": "model",
    "hand_type": "inspire",
    "policy_server_url": "tcp://127.0.0.1:5555",
    # REQ socket RCVTIMEO/SNDTIMEO (ms). First ACT inference can exceed 10s; 0 = wait forever.
    "policy_zmq_timeout_ms": 120_000,
    "h5_path": "PATH_TO_H5",
    "camera_name": "camera",
    "action_rate": 20.0,
    "arm_command": {"mode": "cmd_pos"},
    # Legacy: mode=replay + replay_via_zmq=false is normalized to mode=replay_actions in load_config.
    "replay_via_zmq": False,
    "replay_images_h5_key": None,
    "replay_state_h5_key": None,
    "image_save": {
        "enabled": False,
        "directory": "debug_policy_images",
        "interval": 1,
        "max_frames": 0,
        "use_timestamp_subdir": True,
    },
    "joints": {
        "enabled": False,
        "directory": "debug_joints",
        "use_timestamp_subdir": True,
    },
    # Headless replay_debug only: max ZMQ inference steps (0 = use full HDF5 length).
    "replay_debug_max_steps": 0,
}


def _normalize_arm_command(config: dict) -> dict:
    if "arm_flex_freq_topic" in config:
        raise ValueError(
            "Remove arm_flex_freq_topic from your YAML; arm topics are fixed in ros2_node.py."
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
    if mode not in ARM_CMD_MODES:
        raise ValueError(f"arm_command.mode must be one of {sorted(ARM_CMD_MODES)}, got {mode!r}")
    return {"mode": mode}


def _normalize_run_mode(merged: dict, logger) -> None:
    """Canonical ``mode``: model | replay (HDF5+ZMQ+ROS) | replay_actions (HDF5 actions+ROS) | replay_debug.

    Legacy ``mode: replay`` + ``replay_via_zmq: false`` → ``replay_actions`` (with warning).
    """
    mode = merged.get("mode")
    if mode == "replay":
        if bool(merged.get("replay_via_zmq", False)):
            merged["mode"] = "replay"
        else:
            merged["mode"] = "replay_actions"
            if logger:
                logger.warning(
                    "Deprecated: mode=replay with replay_via_zmq=false — use mode=replay_actions. "
                    "mode=replay now means HDF5 observations + ZMQ + ROS."
                )
    elif mode == "replay_actions" and bool(merged.get("replay_via_zmq", False)) and logger:
        logger.warning("mode=replay_actions ignores replay_via_zmq (open-loop actions only).")


def load_config(config_path: str | None, logger) -> dict:
    # Fallback to the example config when no path is given. Only FileNotFoundError
    # is handled below — a YAML parse error or permission error is treated as
    # misconfiguration and raised to the caller, because silently merging
    # DEFAULT_CONFIG would hide real mistakes (wrong mode, wrong hand_type).
    _bridge_root = Path(__file__).resolve().parent.parent
    if config_path is None:
        config_path = str(_bridge_root / "config" / "config_zmq.example.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Configuration loaded from: {config_path}")
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {}

    merged = {**DEFAULT_CONFIG, **config}
    hand_defaults = HAND_TYPE_DEFAULTS.get(merged["hand_type"], {})
    for k, v in hand_defaults.items():
        merged.setdefault(k, v)

    merged["arm_command"] = _normalize_arm_command(config)
    merged.pop("arm_cmd_mode", None)

    img_def = dict(DEFAULT_CONFIG.get("image_save") or {})
    img_user = config.get("image_save") if isinstance(config.get("image_save"), dict) else {}
    merged["image_save"] = {**img_def, **img_user}

    jt_def = dict(DEFAULT_CONFIG.get("joints") or {})
    jt_user = config.get("joints") if isinstance(config.get("joints"), dict) else {}
    merged["joints"] = {**jt_def, **jt_user}

    # Deep-merge home_hand so that a user who only sets `left` keeps the
    # hand_type default for `right` (setdefault above only fires if the whole
    # home_hand key is missing).
    hh_def = dict(hand_defaults.get("home_hand") or {})
    hh_user = config.get("home_hand") if isinstance(config.get("home_hand"), dict) else {}
    merged["home_hand"] = {**hh_def, **hh_user}

    legacy_jt = config.get("joint_wire_trace")
    if isinstance(legacy_jt, dict):
        if isinstance(config.get("joints"), dict):
            if logger:
                logger.warning("Ignoring deprecated YAML key `joint_wire_trace` because `joints` is set.")
        else:
            if logger:
                logger.warning("YAML key `joint_wire_trace` is deprecated; use `joints` instead.")
            merged["joints"] = {**merged["joints"], **legacy_jt}

    _normalize_run_mode(merged, logger)
    m = merged.get("mode")
    if m not in RUN_MODES:
        raise ValueError(f"Unknown mode {m!r}; expected one of {sorted(RUN_MODES)}")

    return merged


def make_policy_client(config: dict, ros_logger: Any):
    # Deferred import: PolicyClient lives as a sibling in ``robot/``; importing
    # it eagerly at module load would pull ``wire/`` into ``sys.path`` as a
    # side effect of ``policy_client`` loading. Deferring also keeps the
    # replay_debug path cheap when the caller only needs ``load_config``.
    from policy_client import PolicyClient

    url = config["policy_server_url"]
    to_ms = int(config.get("policy_zmq_timeout_ms", 120_000))
    return PolicyClient(
        server_url=url,
        timeout_ms=to_ms,
        ros_logger=ros_logger,
        image_save=config.get("image_save"),
        joints=config.get("joints"),
    )
