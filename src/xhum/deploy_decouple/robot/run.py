#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin entry for the ROS2 + ZMQ deploy node.

Delegates to ``ros2_node_zmq.py``. For ``mode: replay_debug``, we **subprocess**
the same script so ``__name__ == "__main__"`` and the headless path runs before
any ``rclpy`` import (``runpy.run_path`` is not reliable for that on all Python builds).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_ENTRY = _ROOT / "ros2_node_zmq.py"
if not _ENTRY.is_file():
    raise SystemExit(f"missing {_ENTRY}")


def _replay_debug_via_subprocess_if_needed() -> None:
    """If YAML mode is replay_debug, run ros2_node_zmq.py as a real script and exit."""
    if "--config" not in sys.argv:
        return
    try:
        import yaml
    except ImportError:
        return
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=str, default=None)
    args, _ = p.parse_known_args()
    if not args.config:
        return
    cfg = Path(args.config)
    if not cfg.is_file():
        return
    try:
        raw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        return
    if raw.get("mode") != "replay_debug":
        return
    r = subprocess.run(
        [sys.executable, str(_ENTRY), "--config", str(cfg.resolve())],
        cwd=str(_ROOT),
    )
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    try:
        _replay_debug_via_subprocess_if_needed()
    except SystemExit:
        raise
    import runpy

    sys.argv[0] = str(_ENTRY)
    runpy.run_path(str(_ENTRY), run_name="__main__")
