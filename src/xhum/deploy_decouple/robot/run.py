#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single entry for the ROS-side deploy.

Peeks ``mode`` from the YAML and dispatches:
  * ``replay_debug``   → ``replay_debug.run`` (no rclpy import)
  * anything else      → ``ros2_node.main``   (imports rclpy, needs ROS sourced)

Keeping this split here — rather than inside ``ros2_node.py`` — is load-bearing.
Because rclpy imports happen at the top of ``ros2_node.py``, merely *importing*
that module would require ROS sourced, which the replay_debug path on a laptop
specifically wants to avoid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Inject this directory so sibling modules (config_loader, replay_debug,
# ros2_node, policy_client) are importable without needing ``pip install``.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _peek_mode(config_path: str | None) -> str | None:
    """Lightweight YAML peek — avoids pulling in ``config_loader`` (which is
    fine to import here, but defers it to the chosen branch keeps the error
    reporting for bad YAML local to that branch's ``load_config``).

    Returns ``None`` when the file is missing or unreadable; the dispatcher
    then falls through to ``ros2_node.main`` which will surface the real error
    via its usual ``load_config`` path.
    """
    if not config_path:
        return None
    import yaml  # local import: yaml is a runtime dep, not a top-level one

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except OSError:
        return None
    return raw.get("mode") if isinstance(raw, dict) else None


def main() -> None:
    # add_help=False + parse_known_args so we don't intercept --help meant for
    # the ROS node (rclpy accepts its own flags via parse_known_args downstream).
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=str, default=None)
    args, _ = p.parse_known_args()

    if _peek_mode(args.config) == "replay_debug":
        from replay_debug import run as replay_debug_run

        raise SystemExit(replay_debug_run(args.config))

    # Any other mode (model / replay / replay_actions / invalid) → full ROS path.
    from ros2_node import main as ros_main

    ros_main()


if __name__ == "__main__":
    main()
