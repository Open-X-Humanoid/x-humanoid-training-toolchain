# Launch helpers

Thin wrappers so both processes are documented in one place. **Adjust `VENV_*` paths** to match your machine.

| Script | Purpose |
|--------|---------|
| `start_policy.example.sh` | Template: Python 3.12+ env → `policy/policy_server.py`; copy and edit |
| `start_robot.example.sh` | Template: ROS-sourced shell → `robot/run.py --config ...`; copy and edit |
| `start_policy_debug.sh` | Concrete run for the dvt217 debug session; pairs with `robot/config/replay_debug.yaml`. Override `MODEL_PATH=/new/path` to switch checkpoints without editing the file. |

The `.example.sh` files forward `$@` and are meant to be copied. `start_policy_debug.sh` is the tracked, ready-to-run companion of `replay_debug.yaml` — both use `$ROOT/tmp/...` so joints / image dumps from client and server land in the same parent and can be diffed directly.
