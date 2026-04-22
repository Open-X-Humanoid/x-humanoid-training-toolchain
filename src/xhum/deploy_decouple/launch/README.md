# Launch helpers

Thin wrappers so both processes are documented in one place. **Adjust `VENV_*` paths** to match your machine.

| Script | Purpose |
|--------|---------|
| `start_policy.example.sh` | Python 3.12+ env → `policy/policy_server.py` |
| `start_robot.example.sh` | ROS-sourced shell → `robot/run.py --config ...` |

Copy to `start_policy.sh` / `start_robot.sh` and edit paths, or invoke the `python` lines from the root [README](../README.md).
