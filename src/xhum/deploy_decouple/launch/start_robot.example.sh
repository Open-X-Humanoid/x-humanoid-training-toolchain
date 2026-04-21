#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/robot"
# source /opt/ros/humble/setup.bash
exec python3 run.py "$@"
