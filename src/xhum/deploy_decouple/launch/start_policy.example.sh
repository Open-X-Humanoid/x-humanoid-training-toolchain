#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/comms"
# export PATH="/opt/conda/envs/lerobot-0.5.1/bin:$PATH"
# source /path/to/lerobot-venv/bin/activate
export PYTHONPATH="${LEROBOT_SRC:-$(cd "$ROOT/../../.." && pwd)/lerobot/src}:$PYTHONPATH"
exec python policy_server.py "$@"
